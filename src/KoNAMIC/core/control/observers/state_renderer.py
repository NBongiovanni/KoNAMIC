from pathlib import Path
import os
import numpy as np

from KoNAMIC.pipelines.data_pipeline.vision_data.rendering.quad_drawer_2d import QuadDrawer2D
from KoNAMIC.pipelines.data_pipeline.vision_data.rendering.quad_drawer_3d_multi_views import QuadDrawer3DNViews


class StateRenderer2D(QuadDrawer2D):
    def __init__(self, img_size: int):
        super().__init__(img_size, 128)

    def pipeline(self, state: np.ndarray) -> np.ndarray:
        save_dir = Path("results")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        name_fig = Path("raw_images.png")
        img = self._gen_raw_image(
            state,
            save_dir / name_fig,
            True,
            False
        )
        return img


class StateRenderer3D(QuadDrawer3DNViews):
    def __init__(self, img_size: int):
        super().__init__(img_size, 1, 128)

    def pipeline(self, state: np.ndarray):
        save_dir = Path("results")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        imgs = self.render_and_save(
            state,
            save_dir,
            0,
            0,
            True,
        )
        return imgs["left"], imgs["right"]


def make_state_renderer(drone_dim: int):
    if drone_dim == 2:
        return StateRenderer2D(512)
    elif drone_dim == 3:
        return StateRenderer3D(512)
    else:
        raise ValueError(f"Drone dimension inconnue: {drone_dim}")