from __future__ import annotations
from pathlib import Path

from dataclasses import dataclass
from typing import Dict, Tuple, Literal, Optional
import cv2
import numpy as np

ViewId = Literal["left", "right"]

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


class QuadDrawerTwoView2D:
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
        thickness: int = 7,
        views: Optional[Dict[ViewId, ViewSpec]] = None,
    ) -> None:
        self.img_size = int(img_size)
        self.thickness = int(thickness)
        self.save_size = 128

        # Specs par défaut : deux "caméras" latérales (orthographiques) sur x-z et y-z.
        if views is None:
            views = {
                "left": ViewSpec(name="left", axis_h="x", axis_v="z", flip_h=False, flip_v=False),
                "right": ViewSpec(name="right", axis_h="y", axis_v="z", flip_h=False, flip_v=False),
            }
        self.views = views
        self._validate_views()

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
    ) -> Optional[np.ndarray]:
        """
        TODO: Adapt this to the control algorithm.
        Render an image for the given state and save it to disk as PNG.
        """
        img_left, img_right = self.render(state)
        save_path_left = save_path / Path("traj_{}".format(traj_idx)) / Path("left")
        save_path_right = save_path / Path("traj_{}".format(traj_idx)) / Path("right")
        save_path_left.mkdir(parents=True, exist_ok=True)
        save_path_right.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(save_path_left / Path("step_{}.png".format(step_idx))), img_left)
        cv2.imwrite(str(save_path_right /  Path("step_{}.png".format(step_idx))), img_right)
        return [img_left, img_right] if return_img else None

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
        """
        Pipeline complet pour une vue.
        """
        spec = self.views[view_id]
        state2d = self._state3d_to_state2d(state3d, spec)  # (p_h, p_v, alpha)
        segments_local = self._get_body_segments_2d()       # segments en repère local 2D
        segments_world = self._apply_pose_2d(segments_local, state2d)  # pose dans repère vue
        img = self._rasterize_segments(segments_world)
        return self._postprocess(img)

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
    # Étape 1 : state3d -> state2d (par vue)
    # -----------------------------
    @staticmethod
    def _state3d_to_state2d(state3d: np.ndarray, spec: ViewSpec) -> Tuple[float, float, float]:
        """
        Cas découpé : chaque vue dépend d'un seul angle.
          - left  (x-z) : alpha = theta (pitch)
          - right (y-z) : alpha = phi   (roll)

        Args:
            state3d: (6,) = (x, y, z, phi, theta, psi)  (radians)
            spec: ViewSpec pour la vue

        Returns:
            (p_h, p_v, alpha) en float
        """
        state3d = np.asarray(state3d, dtype=np.float32).reshape(-1)
        assert state3d.shape[0] >= 6, f"Expected state3d shape (6,), got {state3d.shape}"

        x, y, z, phi, theta, psi = map(float, state3d[:6])

        # 1) Position par projection orthographique sur le plan de la vue
        axis_map = {"x": x, "y": y, "z": z}
        p_h = axis_map[spec.axis_h]
        p_v = axis_map[spec.axis_v]

        # Offsets / scale éventuels par vue (optionnels mais pratiques)
        p_h = spec.scale * (p_h + spec.offset_h)
        p_v = spec.scale * (p_v + spec.offset_v)

        # Flips éventuels (pour harmoniser orientation left/right)
        if spec.flip_h:
            p_h = -p_h
        if spec.flip_v:
            p_v = -p_v

        # 2) Orientation 2D effective (découplée)
        if spec.name == "left":
            alpha = theta  # pitch "vu" dans x-z
        elif spec.name == "right":
            alpha = phi  # roll "vu" dans y-z
        else:
            raise ValueError(f"Unknown view name: {spec.name}")

        # Si tu veux harmoniser le sens de rotation avec les flips (optionnel)
        # Règle simple : si tu inverses un seul axe, tu inverses l'orientation.
        if spec.flip_h ^ spec.flip_v:
            alpha = -alpha
        return float(p_h), float(p_v), float(alpha)

    # -----------------------------
    # Étape 2 : géométrie canonique 2D (repère corps local)
    # -----------------------------
    @staticmethod
    def _get_body_segments_2d() -> np.ndarray:
        """
        Définit la géométrie canonique 2D du quadrotor en repère corps.

        Returns:
            segments: np.ndarray of shape (S, 2, 2)
                S segments définis par leurs deux extrémités (h, v)
        """
        # -----------------------------
        # Paramètres géométriques
        # -----------------------------
        r_cockpit = 0.05  # demi-côté du cockpit
        r_wing = 0.2  # demi-longueur des ailes

        # -----------------------------
        # Cockpit : carré
        # -----------------------------
        # Sommets du carré (sens anti-horaire)
        p1 = np.array([-r_cockpit, -r_cockpit])
        p2 = np.array([+r_cockpit, -r_cockpit])
        p3 = np.array([+r_cockpit, +r_cockpit])
        p4 = np.array([-r_cockpit, +r_cockpit])
        cockpit_segments = [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]
        # -----------------------------
        # Ailes : segment horizontal
        # -----------------------------
        s1 = np.array([+r_wing, 0.0], dtype=np.float32)
        s3 = np.array([-r_wing, 0.0], dtype=np.float32)

        # “tips” (offset vertical +l2)
        s2 = s1 + np.array([0.0, +r_wing/4], dtype=np.float32)
        s4 = s3 + np.array([0.0, +r_wing/4], dtype=np.float32)

        s5 = s2 + np.array([-r_wing/4, 0.0], dtype=np.float32)
        s6 = s2 + np.array([+r_wing/4, 0.0], dtype=np.float32)
        s7 = s4 + np.array([-r_wing/4, 0.0], dtype=np.float32)
        s8 = s4 + np.array([+r_wing/4, 0.0], dtype=np.float32)

        wing_segments = [(s1, s3), (s1, s2), (s3, s4), (s5, s6), (s7, s8)]

        # -----------------------------
        # Assemblage
        # -----------------------------
        all_segments = cockpit_segments + wing_segments

        # Each segment is defined by two 2D points: (start, end)
        # Convert the list of segments into a single array of shape (S, 2, 2)
        segments = np.empty((len(all_segments), 2, 2), dtype=float)
        for i, (p_start, p_end) in enumerate(all_segments):
            segments[i, 0] = p_start
            segments[i, 1] = p_end
        return segments.astype(np.float32)

    # -----------------------------
    # Étape 3 : pose 2D (translation + rotation)
    # -----------------------------
    @staticmethod
    def _apply_pose_2d(
            segments_local: np.ndarray,
            state2d: Tuple[float, float, float],
    ) -> np.ndarray:
        """
        Applique rotation + translation 2D aux segments locaux.

        Args:
            segments_local: (S, 2, 2) segments en repère corps 2D
            state2d: (p_h, p_v, alpha) pose dans le repère monde de la vue
                - p_h: translation horizontale
                - p_v: translation verticale
                - alpha: rotation (radians), CCW

        Returns:
            segments_world: (S, 2, 2) segments en repère monde de la vue
        """
        segments_local = np.asarray(segments_local, dtype=np.float32)
        assert segments_local.ndim == 3 and segments_local.shape[1:] == (2, 2), \
            f"Expected (S,2,2), got {segments_local.shape}"

        p_h, p_v, alpha = state2d

        # Matrice de rotation 2D (CCW)
        ca = float(np.cos(alpha))
        sa = float(np.sin(alpha))
        R = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)  # (2,2)

        # Rotation : (S,2,2) @ (2,2)^T -> (S,2,2)
        # (on multiplie des vecteurs-lignes [h v], donc on utilise R.T)
        rotated = segments_local @ R.T

        # Translation (broadcast sur S et sur les 2 extrémités)
        t = np.array([p_h, p_v], dtype=np.float32).reshape(1, 1, 2)
        segments_world = rotated + t
        return segments_world

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
            (h1, v1), (h2, v2) = s  # points en monde 2D

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
        px = int(round((p_h + 1) * 0.5 * (self.img_size - 1)))
        py = int(round((1 - (p_v + 1) * 0.5) * (self.img_size - 1)))
        return px, py

    # -----------------------------
    # Étape 5 : postprocess
    # -----------------------------
    def _postprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Inversion / blur / conversion dtype / normalisation.
        """
        img_inv = 255 - img # color inversion
        img_inv = cv2.GaussianBlur(img_inv, (3, 3), sigmaX=0.6)

        img_resized = cv2.resize(
            img_inv,
            (self.save_size, self.save_size),
            interpolation=cv2.INTER_LINEAR
        )
        return cv2.GaussianBlur(img_resized, (3, 3), sigmaX=0.3)

    # -----------------------------
    # Debug / tests unitaires
    # -----------------------------
    def get_segments_world(self, state3d: np.ndarray, view_id: ViewId) -> np.ndarray:
        """
        Utile pour tester la géométrie sans rasterization.
        """
        spec = self.views[view_id]
        state2d = self._state3d_to_state2d(state3d, spec)
        segments_local = self._get_body_segments_2d()
        segments_world = self._apply_pose_2d(segments_local, state2d)
        return segments_world