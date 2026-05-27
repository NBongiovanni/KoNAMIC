from __future__ import annotations
from pathlib import Path

from dataclasses import dataclass
from typing import Dict, Tuple, Literal, Optional
import cv2
import numpy as np

from .drawer_utils import (
    project_segments_to_view,
    clip_segment_to_unit_square,
    postprocess,
    box_segments,
    fill_closed_contours_bw
)

ViewId = Literal["left", "right", "top"]

@dataclass(frozen=True)
class ViewSpec:
    """
    Décrit ce qui distingue une vue de l'autre.
    Pour TwoView2D, une vue est un "plan" (ex: x-z ou y-z) et un mapping de signes/miroirs.
    """
    name: ViewId
    # Choix du plan observé (dans un repère monde 3D) : (axis_h, axis_v)
    # ex: left = ("x", "z"), right = ("y", "z")
    axis_h: Literal["x", "y", "z"]
    axis_v: Literal["x", "y", "z"]

    # Optionnel : miroir horizontal/vertical pour harmoniser l'orientation visuelle
    flip_h: bool = False
    flip_v: bool = False

    # Optionnel : décalage/zoom spécifiques à la vue (si tu veux recadrer différemment)
    scale: float = 1.0
    offset_h: float = 0.0
    offset_v: float = 0.0


class QuadDrawer3D:
    """
    Génère deux vision 2D à partir d'un état 3D (x, y, z, phi, theta, psi),
    via deux vues latérales (ex: plan x-z et plan y-z).

    Architecture (étapes) :
      1) state3d -> state2d(view)  (conversion par vue)
      2) segments canon 2D (repère corps / local)
      3) pose 2D (translation + rotation) dans le repère monde de la vue
      4) rasterization (OpenCV ou autre)
      5) postprocess (invert/blur/resize/normalize)
    """
    def __init__(
            self,
            img_size: int,
            thickness: int,
            views: Optional[Dict[ViewId, ViewSpec]] = None,
    ) -> None:
        self.img_size = int(img_size)
        self.thickness = int(thickness)
        self.save_size = 512

        # Specs par défaut : deux "caméras" latérales (orthographiques) sur x-z et y-z.
        if views is None:
            views = {
                "left": ViewSpec(
                    name="left",
                    axis_h="y",
                    axis_v="z",
                    flip_h=False,
                    flip_v=False
                ),
                "right": ViewSpec(
                    name="right",
                    axis_h="x",
                    axis_v="z",
                    flip_h=True,  # x vers la gauche
                    flip_v=False
                ),
            }
        self.views = views
        self._validate_views()
        self._body_segments_3d = self._get_body_segments_3d()

    # -----------------------------
    # API publique
    # -----------------------------
    def render_and_save(
            self,
            state: np.ndarray,
            save_path: Path,
            traj_idx: int,
            step_idx: int,
            return_img: bool = False,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Render an image for the given state and save it to disk as PNG.
        """
        assert state.ndim == 1 and state.shape[0] >= 6, \
            f"Expected state shape (>=6,), got {state.shape}"

        img_left, img_right = self.render(state[:6])
        traj_dir = save_path / f"traj_{traj_idx}"
        save_path_left = traj_dir / "left" / f"step_{step_idx}.png"
        save_path_right = traj_dir / "right" / f"step_{step_idx}.png"
        save_path_left.parent.mkdir(parents=True, exist_ok=True)
        save_path_right.parent.mkdir(parents=True, exist_ok=True)

        ok_l = cv2.imwrite(str(save_path_left), img_left)
        ok_r = cv2.imwrite(str(save_path_right), img_right)
        assert ok_l and ok_r, "cv2.imwrite failed (path, permissions?)"
        return (img_left, img_right) if return_img else None

    def render(self, state3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            state3d: shape (6,) = (x, y, z, phi, theta, psi)
        Returns:
            img_left, img_right: vision 2D (H, W) float32 ou uint8 selon ton pipeline
        """
        img_left = self.render_view(state3d, "left")
        img_right = self.render_view(state3d, "right")
        return img_left, img_right

    def render_view(self, state3d: np.ndarray, view_id: ViewId) -> np.ndarray:
        segments2d = self._state3d_to_segments2d(state3d, view_id)  # (S,2,2)
        img = self._rasterize_segments(segments2d)
        img = fill_closed_contours_bw(img, close_gaps=True, close_ksize=3)
        return postprocess(img, self.save_size)

    # -----------------------------
    # Étape 0 : validation / conventions
    # -----------------------------
    def _validate_views(self) -> None:
        # Vérifie que les vues existent et que les axes sont distincts.
        required = {"left", "right"}
        if set(self.views.keys()) != required:
            raise ValueError(f"views must have exactly keys {required}, got {set(self.views.keys())}")

        for k, spec in self.views.items():
            if spec.axis_h == spec.axis_v:
                raise ValueError(f"View {k}: axis_h and axis_v must be different (got {spec.axis_h}).")

    # -----------------------------
    # Étape 2 : géométrie canonique 2D (repère corps local)
    # -----------------------------
    @staticmethod
    def _get_body_segments_3d() -> np.ndarray:
        """
        Géométrie canonique 3D du quadrotor en repère corps.

        Returns:
            segments: np.ndarray of shape (S, 2, 3)
                S segments définis par leurs deux extrémités (x, y, z)
        """
        # -----------------------------
        # Paramètres géométriques
        # -----------------------------

        z0 = 0.0
        # -----------------------------
        # Cockpit : "boîte" (épaisseur en z)
        # -----------------------------
        r_cockpit = 0.03
        h_cockpit = 0.03  # épaisseur (à ajuster)

        z_top = +0.5 * h_cockpit
        z_bot = -0.5 * h_cockpit

        # Carré du haut
        c1t = np.array([-r_cockpit, -r_cockpit, z_top], dtype=np.float32)
        c2t = np.array([+r_cockpit, -r_cockpit, z_top], dtype=np.float32)
        c3t = np.array([+r_cockpit, +r_cockpit, z_top], dtype=np.float32)
        c4t = np.array([-r_cockpit, +r_cockpit, z_top], dtype=np.float32)

        # Carré du bas
        c1b = np.array([-r_cockpit, -r_cockpit, z_bot], dtype=np.float32)
        c2b = np.array([+r_cockpit, -r_cockpit, z_bot], dtype=np.float32)
        c3b = np.array([+r_cockpit, +r_cockpit, z_bot], dtype=np.float32)
        c4b = np.array([-r_cockpit, +r_cockpit, z_bot], dtype=np.float32)

        cockpit_segments = [
            # contour haut
            (c1t, c2t), (c2t, c3t), (c3t, c4t), (c4t, c1t),
            # contour bas
            (c1b, c2b), (c2b, c3b), (c3b, c4b), (c4b, c1b),
            # arêtes verticales
            (c1b, c1t), (c2b, c2t), (c3b, c3t), (c4b, c4t),
        ]

        # -----------------------------
        # Ailes : "croix" (branche X et branche Y)
        # -----------------------------
        r_wing = 0.1  # demi-longueur
        wing_w = 0.03  # largeur (épaisseur en Y pour le bras X, ou en X pour le bras Y)
        wing_h = 0.02  # épaisseur en Z

        # Bras X : long en X, fin en Y, fin en Z
        arm_x = box_segments(
            center=np.array([0.0, 0.0, 0.0], np.float32),
            size_xyz=np.array([2 * r_wing, wing_w, wing_h], np.float32),
        )

        # Bras Y : long en Y, fin en X, fin en Z
        arm_y = box_segments(
            center=np.array([0.0, 0.0, 0.0], np.float32),
            size_xyz=np.array([wing_w, 2 * r_wing, wing_h], np.float32),
        )
        wing_segments = arm_x + arm_y

        # -----------------------------
        # Assemblage
        # -----------------------------
        all_segments = cockpit_segments + wing_segments
        segments = np.empty((len(all_segments), 2, 3), dtype=np.float32)
        for i, (p_start, p_end) in enumerate(all_segments):
            segments[i, 0] = p_start
            segments[i, 1] = p_end
        return segments

    # -----------------------------
    # Étape 3 : pose 2D (translation + rotation)
    # -----------------------------
    @staticmethod
    def _apply_pose_3d(points_or_segments: np.ndarray, state3d: np.ndarray) -> np.ndarray:
        """
        Applique la pose 3D (rotation + translation) à des points 3D ou à des segments 3D.

        Args:
            points_or_segments:
                - soit array de points de shape (N, 3)
                - soit array de segments de shape (S, 2, 3)
            state3d: array de shape (6,) contenant (x, y, z, phi, theta, psi)
                    avec phi=roll, theta=pitch, psi=yaw en radians.

        Returns:
            transformed: array de même shape que points_or_segments, en repère "monde".
        """
        assert state3d.shape[0] == 6, f"Expected state3d shape (6,), got {state3d.shape}"

        x, y, z, phi, theta, psi = state3d.astype(np.float32)

        # --- Matrices de rotation (convention Z-Y-X : yaw -> pitch -> roll) ---
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)

        Rx = np.array([[1.0, 0.0, 0.0],
                       [0.0, cphi, -sphi],
                       [0.0, sphi, cphi]], dtype=np.float32)

        Ry = np.array([[cth, 0.0, sth],
                       [0.0, 1.0, 0.0],
                       [-sth, 0.0, cth]], dtype=np.float32)

        Rz = np.array([[cpsi, -spsi, 0.0],
                       [spsi, cpsi, 0.0],
                       [0.0, 0.0, 1.0]], dtype=np.float32)

        R = Rz @ Ry @ Rx  # rotation corps->monde

        t = np.array([x, y, z], dtype=np.float32)

        arr = points_or_segments.astype(np.float32, copy=False)

        # --- Cas points (N,3) ---
        if arr.ndim == 2 and arr.shape[-1] == 3:
            # (N,3) -> (N,3)
            return (arr @ R.T) + t

        # --- Cas segments (S,2,3) ---
        if arr.ndim == 3 and arr.shape[-2:] == (2, 3):
            # On "flatten" en (S*2,3), on transforme, puis reshape
            flat = arr.reshape(-1, 3)
            flat_w = (flat @ R.T) + t
            return flat_w.reshape(arr.shape)

        raise ValueError(
            f"Unsupported shape for points_or_segments: {arr.shape}. "
            "Expected (N,3) or (S,2,3)."
        )

    # -----------------------------
    # Étape 4 : rasterization (segments -> image)
    # -----------------------------
    def _rasterize_segments(self, segments_world: np.ndarray) -> np.ndarray:
        """
        Rasterize des segments 2D en image (H, W).

        Args:
            segments_world: (S, 2, 2) avec coordonnées (h, v) dans le repère monde de la vue.
        Returns:
            img: (H, W) uint8, fond noir, segments blancs.
        """
        segments_world = np.asarray(segments_world, dtype=np.float32)
        assert segments_world.ndim == 3 and segments_world.shape[1:] == (2, 2), \
            f"Expected (S,2,2), got {segments_world.shape}"

        H = W = self.img_size
        img = np.zeros((H, W), dtype=np.uint8)

        # Pour éviter les problèmes si des NaN/inf apparaissent (rare mais ça arrive vite)
        # mask shape: (S,) True si les 4 scalaires sont finis
        finite_mask = np.isfinite(segments_world).all(axis=(1, 2))
        segs = segments_world[finite_mask]

        # Trace chaque segment
        for s in segs:
            p0 = s[0]  # (h,v)
            p1 = s[1]

            clipped = clip_segment_to_unit_square(p0, p1)
            if clipped is None:
                continue
            (h1, v1), (h2, v2) = clipped

            x1, y1 = self._to_pixel(float(h1), float(v1))
            x2, y2 = self._to_pixel(float(h2), float(v2))

            cv2.line(
                img,
                (x1, y1),
                (x2, y2),
                color=255,
                thickness=self.thickness,
                lineType=cv2.LINE_8,  # binaire, net (mets LINE_AA si tu veux des bords lissés)
            )
        return img

    def _to_pixel(self, p_h: float, p_v: float) -> Tuple[int, int]:
        """
        Mapping monde 2D (h,v) -> pixels (u,v) dans [0..W-1],[0..H-1]
        """
        # px = int(round((p_h + 1) * 0.5 * (self.img_size - 1)))
        # py = int(round((1 - (p_v + 1) * 0.5) * (self.img_size - 1)))
        px = int(round((p_h + 0.5) * (self.img_size - 1)))
        py = int(round((0.5 - p_v) * (self.img_size - 1)))
        return px, py


    def _state3d_to_segments2d(self, state3d: np.ndarray, view_id: ViewId) -> np.ndarray:
        """
        Nouveau remplaçant conceptuel de _state3d_to_state2d():
        retourne directement les segments 2D projetés pour la vue demandée.

        Returns:
            segments2d: (S, 2, 2)
        """
        # 2) pose 3D (corps -> monde)
        body_segments_world = self._apply_pose_3d(self._body_segments_3d, state3d)  # (S,2,3)

        # 3) projection par vue
        view_spec = self.views[view_id]  # ou self.view_specs[view_id] selon ton code
        segments2d = project_segments_to_view(body_segments_world, view_spec)  # (S,2,2)
        return segments2d

