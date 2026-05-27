from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Literal, Mapping

import cv2
import numpy as np

# Reprend tes utilitaires existants
from .drawer_utils import (
    project_segments_to_view,
    clip_segment_to_unit_square,
    postprocess,
    box_segments,
    fill_closed_contours_bw
)

Axis = Literal["x", "y", "z"]
ViewIdN = str


@dataclass(frozen=True)
class ViewSpecN:
    """
    Spécification d'une vue 2D orthographique définie par:
      - un plan (axis_h, axis_v) extrait du 3D monde
      - flips (pour l'orientation visuelle)
      - scale + offsets (pour cadrer/normaliser)
    """
    name: ViewIdN
    axis_h: Axis
    axis_v: Axis
    flip_h: bool = False
    flip_v: bool = False
    scale: float = 1.0
    offset_h: float = 0.0
    offset_v: float = 0.0


class QuadDrawer3DNViews:
    """
    Version N-vues du QuadDrawer3D.

    Points clés:
      - self.views: dict[str, ViewSpecN]
      - render(state3d) -> dict[view_id, image]
      - render_and_save(...) boucle sur toutes les vues et sauvegarde dans traj_{i}/{view_id}/step_{k}.png

    Hypothèse: tu gardes tes conventions de clipping et mapping pixel telles quelles.
    """
    def __init__(
        self,
        img_size: int,
        thickness: int,
        save_size: int,
        views: Optional[Dict[ViewIdN, ViewSpecN]] = None,
    ) -> None:
        self.img_size = int(img_size)
        self.thickness = int(thickness)
        self.save_size = int(save_size)

        # Par défaut: 2 latérales + 1 top
        # NB: tes "left/right" par défaut utilisent (y,z) et (x,z) avec flip_h sur right.
        if views is None:
            views = {
                "left": ViewSpecN(
                    name="left",
                    axis_h="y",
                    axis_v="z",
                    flip_h=False,
                    flip_v=False,
                ),
                "right": ViewSpecN(
                    name="right",
                    axis_h="x",
                    axis_v="z",
                    flip_h=False,   # x vers la gauche (comme dans ton code actuel)
                    flip_v=False,
                ),
                "top": ViewSpecN(
                    name="top",
                    axis_h="x",
                    axis_v="y",
                    flip_h=False,
                    flip_v=True,   # souvent utile pour avoir +y "vers le haut" de l'image
                ),
            }

        self.views: Dict[ViewIdN, ViewSpecN] = dict(views)
        self._validate_views()

        # Géométrie canonique 3D du quad (reprends ton implémentation existante)
        self._body_segments_3d = self._get_body_segments_3d()

    # -----------------------------
    # API publique
    # -----------------------------
    def render(self, state3d: np.ndarray) -> Dict[ViewIdN, np.ndarray]:
        """
        Args:
            state3d: shape (6,) = (x, y, z, phi, theta, psi)
        Returns:
            dict {view_id: img}
        """
        assert state3d.ndim == 1 and state3d.shape[0] >= 6, \
            f"Expected state shape (>=6,), got {state3d.shape}"

        out: Dict[ViewIdN, np.ndarray] = {}
        for view_id in self.views.keys():
            out[view_id] = self.render_view(state3d[:6], view_id)
        return out

    def render_view(self, state3d: np.ndarray, view_id: ViewIdN) -> np.ndarray:
        segments2d = self._state3d_to_segments2d(state3d, view_id)  # (S,2,2)
        img = self._rasterize_segments(segments2d)
        img = fill_closed_contours_bw(img, close_gaps=True, close_ksize=3)
        return postprocess(img, self.save_size)

    def render_and_save(
        self,
        state: np.ndarray,
        save_path: Path,
        traj_idx: int,
        step_idx: int,
        return_img: bool = False,
    ) -> Optional[Dict[ViewIdN, np.ndarray]]:
        """
        Sauvegarde:
          save_path/traj_{traj_idx}/{view_id}/step_{step_idx}.png
        """
        imgs = self.render(state)

        traj_dir = save_path / f"traj_{traj_idx}"
        for view_id, img in imgs.items():
            p = traj_dir / view_id / f"step_{step_idx}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(str(p), img)
            assert ok, f"cv2.imwrite failed for {p} (path, permissions?)"

        return imgs if return_img else None

    # -----------------------------
    # Validation / conventions
    # -----------------------------
    def _validate_views(self) -> None:
        if not isinstance(self.views, Mapping) or len(self.views) == 0:
            raise ValueError("views must be a non-empty dict[str, ViewSpecN].")

        for k, spec in self.views.items():
            if k != spec.name:
                raise ValueError(f"View key '{k}' must match spec.name='{spec.name}'.")
            if spec.axis_h == spec.axis_v:
                raise ValueError(
                    f"View '{k}': axis_h and axis_v must be different (got {spec.axis_h})."
                )
            if spec.axis_h not in ("x", "y", "z") or spec.axis_v not in ("x", "y", "z"):
                raise ValueError(f"View '{k}': axis must be in {{'x','y','z'}}.")

    # -----------------------------
    # Pipeline interne (reprend ta logique)
    # -----------------------------
    def _state3d_to_segments2d(self, state3d: np.ndarray, view_id: ViewIdN) -> np.ndarray:
        body_segments_world = self._apply_pose_3d(
            self._body_segments_3d,
            state3d
        )  # (S,2,3)
        view_spec = self.views[view_id]
        segments2d = project_segments_to_view(
            body_segments_world,
            view_spec
        )  # (S,2,2)
        return segments2d

    def _rasterize_segments(self, segments_world: np.ndarray) -> np.ndarray:
        segments_world = np.asarray(segments_world, dtype=np.float32)
        assert segments_world.ndim == 3 and segments_world.shape[1:] == (2, 2), \
            f"Expected (S,2,2), got {segments_world.shape}"

        H = W = self.img_size
        img = np.zeros((H, W), dtype=np.uint8)

        finite_mask = np.isfinite(segments_world).all(axis=(1, 2))
        segs = segments_world[finite_mask]

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
                lineType=cv2.LINE_8,
            )
        return img

    def _to_pixel(self, p_h: float, p_v: float) -> tuple[int, int]:
        # mapping identique à ton code actuel (monde 2D dans [-0.5,0.5])
        px = int(round((p_h + 0.5) * (self.img_size - 1)))
        py = int(round((0.5 - p_v) * (self.img_size - 1)))
        return px, py

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