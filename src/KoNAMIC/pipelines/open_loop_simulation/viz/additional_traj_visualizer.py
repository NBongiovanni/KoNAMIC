from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib import rc_context

from KoNAMIC.core.drone import get_x_labels
from KoNAMIC.viz.primitives_single import plot_x_gt
from KoNAMIC.viz.style import DEFAULT_RC_PARAMS, save_figure

GetArray = Callable[[Any], np.ndarray]
GetOptArray = Callable[[Any], Optional[np.ndarray]]

RAD_TO_DEG = 180.0 / np.pi

@dataclass(frozen=True)
class SinglePlotExtractors:
    """
    Defines how to extract arrays to plot from an arbitrary rollout output object.
    All arrays are expected to be numpy arrays.
    """
    get_x_gt: GetArray
    get_x_pred: GetOptArray
    get_u: GetArray


class AdditionalTrajVisualizer:
    """
    Plot ONE rollout (single trajectory):
      - sensor: x_gt vs x_pred (aligned)
      - inputs: u

    The class is modality-agnostic. You inject extractors at init.
    """
    WIDTH_FIGURES: float = 5.5
    HEIGHT_FIGURES: float = 1.0

    def __init__(
            self,
            drone_dim: int,
            dt: float,
            path: Path,
            x_gt: np.ndarray,
            only_position: bool,
    ) -> None:
        assert drone_dim in (2, 3)

        self.drone_dim = drone_dim
        self.dt = dt
        self.path = path
        self.x_gt = x_gt
        self.only_position = only_position
        num_steps = x_gt.shape[0]
        self.time = np.arange(num_steps) * self.dt

        self.rc_params = DEFAULT_RC_PARAMS
        self.output: Any = None

    # -------------------------
    # Public API
    # -------------------------
    def pipeline(self, output: Any) -> None:
        """
        Run a full visualization pipeline for a single trajectory.
        """
        self.output = output
        save_dir = self.path
        save_dir.mkdir(parents=True, exist_ok=True)

        with rc_context(self.rc_params):
            self.plot_states(save_dir)

    # -------------------------
    # States
    # -------------------------
    def plot_states(self, save_dir: Path) -> None:
        x_labels = get_x_labels(self.drone_dim, self.only_position)
        if self.drone_dim == 3:
            x_dim_disp = 6
        elif self.drone_dim == 2:
            x_dim_disp = 3
        else:
            raise ValueError("Unknown drone dimension")

        fig, axes = plt.subplots(
            x_dim_disp,
            1,
            figsize=(self.WIDTH_FIGURES, x_dim_disp * self.HEIGHT_FIGURES),
            constrained_layout=True,
            squeeze=False,
        )

        for i in range(x_dim_disp):
            ax: Axes = axes[i, 0]
            plot_x_gt(
                [ax],
                self.time,
                [x_labels[i]],
                self.x_gt[:, i:i + 1],
                True,
            )

        save_figure(fig, save_dir, "sensor.pdf")