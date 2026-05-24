from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import cv2

class QuadDrawer2D:
    """
    Responsible only for rendering a quadrotor image from a given state.

    Public API:
    - render(state) -> np.ndarray
    - render_and_save(state, save_path) -> Optional[np.ndarray]
    """
    def __init__(self, img_size: int, save_size: int) -> None:
        """
        Parameters
        ----------
        img_size : int
            Internal canvas size used for drawing (e.g. 128).
        save_size : int
            Final size of the saved image (can be equal or different from img_size).
        """
        self.img_size = img_size
        self.save_size = save_size

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------
    def _gen_raw_image(
            self,
            state: np.ndarray,
            save_path: Path,
            return_img=False,
            save_im=True
    ) -> Optional[np.ndarray]:

        img = self._render_image(state)
        if save_im:
            cv2.imwrite(str(save_path), img)
        return img if return_img else None

    def render(self, state: np.ndarray) -> np.ndarray:
        """
        Render a single grayscale image corresponding to the provided state.

        Parameters
        ----------
        state : np.ndarray
            Array of shape (6,) or (1,6) containing [y0, z0, theta, ...].

        Returns
        -------
        img : np.ndarray
            2D uint8 array of shape (save_size, save_size), values in [0,255].
            Object is white on a black background.
        """
        return self._render_image(state)

    def render_and_save(
            self,
            state: np.ndarray,
            save_path: Path,
            traj_idx: int,
            step_idx: int,
            return_img: bool = False
    ) -> Optional[np.ndarray]:
        """
        Render an image for the given state and save it to disk as PNG.
        """
        traj_dir = save_path / f"traj_{traj_idx}"
        save_path_final = traj_dir / f"step_{step_idx}.png"
        traj_dir.parent.mkdir(parents=True, exist_ok=True)

        img = self._render_image(state)
        cv2.imwrite(str(save_path_final), img)
        return img if return_img else None

    # ------------------------------------------------------------------
    # Internal helpers (private)
    # ------------------------------------------------------------------
    def _render_image(self, state: np.ndarray) -> np.ndarray:
        """
        Low-level rendering function.
        """
        # 1) Fond blanc
        img = np.ones((self.img_size, self.img_size), dtype=np.uint8) * 255

        # 2) Coordonnées géométriques dans [-1,1]×[-1,1]
        wings_coord = self._compute_coordinates_wings(state)
        s5 = wings_coord[4]
        s6 = wings_coord[5]
        cockpit_coord = self._compute_coordinates_cockpit(state, s5, s6)

        # 3) Tracé des segments (en noir)
        self._plot_quad(img, wings_coord, cockpit_coord)

        # 4) Inversion des couleurs → objet blanc sur fond noir
        img_inv = 255 - img

        # 5) Blur léger pour aider la robustesse
        img_inv = cv2.GaussianBlur(img_inv, (3, 3), sigmaX=0.6)

        # 6) Redimensionnement en save_size×save_size si nécessaire
        if self.save_size != self.img_size:
            img_resized = cv2.resize(
                img_inv,
                (self.save_size, self.save_size),
                interpolation=cv2.INTER_LINEAR
            )
        else:
            img_resized = img_inv

        # 7) Blur très léger final
        img_out = cv2.GaussianBlur(img_resized, (3, 3), sigmaX=0.3)
        return img_out

    @staticmethod
    def _compute_coordinates_wings(state: np.ndarray):
        # ... (exactement ton code actuel)
        y0, z0, theta = state.ravel()[:3]
        l1 = 0.2
        l2 = l1 / 4
        l4 = l1 / 6

        s1 = [y0 + l1 * np.cos(theta), z0 + l1 * np.sin(theta)]
        s2 = [s1[0] + l2 * np.cos(theta + np.pi/2),
              s1[1] + l2 * np.sin(theta + np.pi/2)]

        s3 = [y0 + l1 * np.cos(theta + np.pi),
              z0 + l1 * np.sin(theta + np.pi)]
        s4 = [s3[0] + l2 * np.cos(theta + np.pi/2),
              s3[1] + l2 * np.sin(theta + np.pi/2)]

        s5 = [y0 + l2 * np.cos(theta + np.pi),
              z0 + l2 * np.sin(theta + np.pi)]
        s6 = [y0 + l2 * np.cos(theta),
              z0 + l2 * np.sin(theta)]

        s7 = [s2[0] + l4 * np.cos(theta),
              s2[1] + l4 * np.sin(theta)]
        s8 = [s2[0] + l4 * np.cos(theta + np.pi),
              s2[1] + l4 * np.sin(theta + np.pi)]

        s9  = [s4[0] + l4 * np.cos(theta),
               s4[1] + l4 * np.sin(theta)]
        s10 = [s4[0] + l4 * np.cos(theta + np.pi),
               s4[1] + l4 * np.sin(theta + np.pi)]
        return s1, s2, s3, s4, s5, s6, s7, s8, s9, s10

    @staticmethod
    def _compute_coordinates_cockpit(state: np.ndarray, s5, s6):
        # ... (ton code actuel)
        theta = state.ravel()[2]
        l1 = 0.2
        l2 = l1 / 4

        c1  = [s5[0] + l2 * np.cos(theta - np.pi/2),
               s5[1] + l2 * np.sin(theta - np.pi/2)]
        c2  = [s6[0] + l2 * np.cos(theta - np.pi/2),
               s6[1] + l2 * np.sin(theta - np.pi/2)]
        c3  = [s6[0] + l2 * np.cos(theta + np.pi/2),
               s6[1] + l2 * np.sin(theta + np.pi/2)]
        c4  = [s5[0] + l2 * np.cos(theta + np.pi/2),
               s5[1] + l2 * np.sin(theta + np.pi/2)]

        c12 = [s5[0] + l2/3 * np.cos(theta - np.pi/2),
               s5[1] + l2/2 * np.sin(theta - np.pi/2)]
        c22 = [s6[0] + l2/3 * np.cos(theta - np.pi/2),
               s6[1] + l2/2 * np.sin(theta - np.pi/2)]
        c32 = [s6[0] + l2/3 * np.cos(theta + np.pi/2),
               s6[1] + l2/2 * np.sin(theta + np.pi/2)]
        c42 = [s5[0] + l2/3 * np.cos(theta + np.pi/2),
               s5[1] + l2/2 * np.sin(theta + np.pi/2)]
        return c1, c2, c3, c4, c12, c22, c32, c42

    def _to_pixel(self, pt) -> tuple[int, int]:
        """
        Transform a point from normalized coords [-1,1]^2 to pixel coordinates.
        """
        x_norm, y_norm = pt
        px = int(round((x_norm + 1) * 0.5 * (self.img_size - 1)))
        py = int(round((1 - (y_norm + 1) * 0.5) * (self.img_size - 1)))
        return px, py

    def _plot_quad(
            self,
            img: np.ndarray,
            wings_coord: tuple,
            cockpit_coord: tuple
    ) -> None:
        """
        Draw the quadrotor on the given image (in-place).
        """
        s1, s2, s3, s4, s5, s6, s7, s8, s9, s10 = wings_coord
        c1, c2, c3, c4, c12, c22, c32, c42 = cockpit_coord

        lw = 7  # line width

        def draw(p, q, color: Sequence[int] = (0,), thickness: int = lw) -> None:
            cv2.line(
                img,
                self._to_pixel(p),
                self._to_pixel(q),
                color,
                thickness
            )

        # wings
        draw(s1, s3)
        draw(s1, s2)
        draw(s3, s4)
        draw(s7, s8, thickness=lw // 2)
        draw(s9, s10, thickness=lw // 2)

        # cockpit
        draw(c2, c3)
        draw(c1, c4)
        draw(c3, c4)
        draw(c1, c2)
        draw(c32, c42, thickness=lw // 2)
        draw(c12, c22, thickness=lw // 2)