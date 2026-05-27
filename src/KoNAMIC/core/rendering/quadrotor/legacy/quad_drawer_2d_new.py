from pathlib import Path

import numpy as np
import cv2

PI = np.pi

class QuadDrawer2DNew:
    """
    Responsible only for rendering a quadrotor image from a given state.
    """
    def __init__(self, img_size: int) -> None:
        self.img_size = img_size
        self.save_size = 128
        # Geometry (canonical, in normalized coords)
        self.l1 = 0.2
        self.l2 = self.l1 / 4
        self.thickness = 7

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------
    def render_and_save(self, state: np.ndarray, save_path: Path, return_img: bool = False):
        """
        Render a single grayscale image corresponding to the provided state.
        The parameter return_image is used in the control loop
        """
        img = self._render_image(state)
        cv2.imwrite(str(save_path), img)
        return img if return_img else None

    # ------------------------------------------------------------------
    # Internal helpers (private)
    # ------------------------------------------------------------------
    def _render_image(self, state: np.ndarray) -> np.ndarray:
        """
        Low-level rendering function.
        """
        img = np.ones((self.img_size, self.img_size), dtype=np.uint8) * 255 # white background
        segments = self._get_body_segments_2d(state)
        self._rasterize_segments(img, segments)
        return self._postprocess(img)

    def _to_pixel(self, point: tuple) -> tuple[int, int]:
        """
        Transform a point from normalized coords [-1,1]^2 to pixel coordinates.
        """
        x_normalised, y_normalised = point
        x_pixel = int(round((x_normalised + 1) * 0.5 * (self.img_size - 1)))
        y_pixel = int(round((1 - (y_normalised + 1) * 0.5) * (self.img_size - 1)))
        return x_pixel, y_pixel

    def _get_body_segments_2d(self, state: np.ndarray) -> np.ndarray:
        """
        Construit les segments monde à partir de la géométrie canonique,
        puis applique la pose (y0,z0,theta).
        """
        y0, z0, theta = np.asarray(state, dtype=np.float32).ravel()[:3]
        segments_local = self._get_canonical_segments_2d()
        segments_world = self._apply_pose_2d(segments_local, y0=y0, z0=z0, theta=theta)
        return segments_world

    def _rasterize_segments(self, img: np.ndarray, segments: np.ndarray) -> None:
        """
        Trace des segments sur img (in-place), en noir, en réutilisant _to_pixel.
        """
        for (p, q) in segments:
            cv2.line(
                img,
                self._to_pixel(p),
                self._to_pixel(q),
                (0,),
                self.thickness
            )

    def _get_canonical_segments_2d(self) -> np.ndarray:
        """
        Segments du quad en repère local (theta=0), centré en (0,0).
        Convention: axe h = y, axe v = z.

        Retour:
            segments_local: (S, 2, 2) float32
        """
        l1, l2 = self.l1, self.l2

        # --- Wings (theta=0 => direction +h) ---
        # s1 = (+l1, 0), s3 = (-l1, 0)
        s1 = np.array([+l1, 0.0], dtype=np.float32)
        s3 = np.array([-l1, 0.0], dtype=np.float32)

        # “tips” (offset vertical +l2)
        s2 = s1 + np.array([0.0, +l2], dtype=np.float32)
        s4 = s3 + np.array([0.0, +l2], dtype=np.float32)

        # --- Cockpit ---
        # Dans ton ancien code, s5 et s6 sont sur l’axe h à +/- l2
        s5 = np.array([-l2, 0.0], dtype=np.float32)
        s6 = np.array([+l2, 0.0], dtype=np.float32)

        # Carré cockpit: offset vertical +/- l2
        c1 = s5 + np.array([0.0, -l2], dtype=np.float32)
        c2 = s6 + np.array([0.0, -l2], dtype=np.float32)
        c3 = s6 + np.array([0.0, +l2], dtype=np.float32)
        c4 = s5 + np.array([0.0, +l2], dtype=np.float32)

        # Traits internes (demi-l2 en vertical, comme ton code actuel)
        c12 = s5 + np.array([0.0, -l2 / 2], dtype=np.float32)
        c22 = s6 + np.array([0.0, -l2 / 2], dtype=np.float32)
        c32 = s6 + np.array([0.0, +l2 / 2], dtype=np.float32)
        c42 = s5 + np.array([0.0, +l2 / 2], dtype=np.float32)

        seg = [
            # Wings
            (s1, s3),
            (s1, s2),
            (s3, s4),

            # Cockpit outer box
            (c2, c3),
            (c1, c4),
            (c3, c4),
            (c1, c2),

            # Cockpit inner lines
            (c32, c42),
            (c12, c22),
        ]

        return np.array([[p, q] for (p, q) in seg], dtype=np.float32)

    @staticmethod
    def _apply_pose_2d(
            segments_local: np.ndarray,
            y0: float,
            z0: float,
            theta: float,
    ) -> np.ndarray:
        """
        Applique rotation + translation à des segments (S,2,2).
        Rotation CCW dans le plan (y,z).
        """
        segments_local = np.asarray(segments_local, dtype=np.float32)
        assert segments_local.ndim == 3 and segments_local.shape[1:] == (2, 2), \
            f"Expected (S,2,2), got {segments_local.shape}"

        ca = float(np.cos(theta))
        sa = float(np.sin(theta))
        R = np.array([[ca, -sa],[sa, ca]], dtype=np.float32)

        rotated = segments_local @ R.T
        t = np.array([y0, z0], dtype=np.float32).reshape(1, 1, 2)
        return rotated + t

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
