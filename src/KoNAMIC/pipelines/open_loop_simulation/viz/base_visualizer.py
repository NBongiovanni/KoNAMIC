from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc_context
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from KoNAMIC.viz.style import DEFAULT_RC_PARAMS


class BaseOpenLoopVisualizer(ABC):
    WIDTH_FIGURES: float = 5.5
    HEIGHT_FIGURES: float = 1.0

    def __init__(
        self,
        system_dim: int,
        dt: float,
        only_position: bool,
        num_columns_states: int,
        num_columns_inputs: int,
        rc_params: Optional[dict] = None,
    ) -> None:
        assert system_dim in (2, 3)
        assert num_columns_states in (1, 2)
        assert num_columns_inputs in (1, 2)

        self.drone_dim = int(system_dim)
        self.dt = float(dt)
        self.only_position = bool(only_position)
        self.num_columns_states = int(num_columns_states)
        self.num_columns_inputs = int(num_columns_inputs)
        self.rc_params = DEFAULT_RC_PARAMS if rc_params is None else rc_params

    # -------------------------
    # Public pipeline
    # -------------------------
    def run_with_rc_context(self) -> None:
        with rc_context(self.rc_params):
            self._plot()

    @abstractmethod
    def _plot(self) -> None:
        """
        Implemented by subclasses.
        """
        raise NotImplementedError

    # -------------------------
    # Layout helpers
    # -------------------------
    @staticmethod
    def _compute_grid_shape(n_plots: int, n_cols: int) -> tuple[int, int]:
        n_rows = int(np.ceil(n_plots / n_cols))
        return n_rows, n_cols

    @staticmethod
    def _build_block_axes(
        fig: Figure,
        gs: gridspec.GridSpec,
        start_row: int,
        n_plots: int,
        n_rows: int,
        n_cols: int,
    ) -> tuple[npt.NDArray[object], list[Axes]]:
        axes_grid = np.full((n_rows, n_cols), None, dtype=object)
        axes_used: list[Axes] = []

        for k in range(n_plots):
            if n_cols == 1:
                row = k
                col = 0
            elif n_cols == 2:
                row = k % n_rows
                col = k // n_rows
            else:
                raise NotImplementedError("Only 1 or 2 columns are supported.")

            gs_row = start_row + row

            if n_cols == 1:
                ax = fig.add_subplot(gs[gs_row, :])
            else:
                ax = fig.add_subplot(gs[gs_row, col])

            axes_grid[row, col] = ax
            axes_used.append(ax)

        return axes_grid, axes_used

    def _make_states_inputs_layout(
        self,
        n_states: int,
        n_inputs: int,
    ) -> tuple[Figure, npt.NDArray[object], list[Axes], npt.NDArray[object], list[Axes]]:
        n_state_rows, n_state_cols = self._compute_grid_shape(
            n_states,
            self.num_columns_states,
        )
        n_input_rows, n_input_cols = self._compute_grid_shape(
            n_inputs,
            self.num_columns_inputs,
        )

        total_rows = n_state_rows + n_input_rows

        fig = plt.figure(
            figsize=(self.WIDTH_FIGURES, total_rows * self.HEIGHT_FIGURES),
            constrained_layout=True,
        )

        gs = gridspec.GridSpec(total_rows, ncols=2, figure=fig)

        state_axes_grid, state_axes_used = self._build_block_axes(
            fig=fig,
            gs=gs,
            start_row=0,
            n_plots=n_states,
            n_rows=n_state_rows,
            n_cols=n_state_cols,
        )

        input_axes_grid, input_axes_used = self._build_block_axes(
            fig=fig,
            gs=gs,
            start_row=n_state_rows,
            n_plots=n_inputs,
            n_rows=n_input_rows,
            n_cols=n_input_cols,
        )

        return fig, state_axes_grid, state_axes_used, input_axes_grid, input_axes_used

    # -------------------------
    # Optional xlabel helpers
    # -------------------------
    @staticmethod
    def _hide_inner_xlabels_block(
        axes_grid: npt.NDArray[object],
        n_cols: int,
    ) -> None:
        n_rows = axes_grid.shape[0]
        for row in range(n_rows - 1):
            for col in range(n_cols):
                ax = axes_grid[row, col]
                if ax is not None:
                    ax.set_xlabel("")

    @staticmethod
    def _keep_only_bottom_xlabel(
        input_axes_grid: npt.NDArray[object],
        xlabel: str = "Time [s]",
    ) -> None:
        for ax in input_axes_grid.flatten():
            if ax is not None:
                ax.set_xlabel("")

        bottom_row = input_axes_grid.shape[0] - 1
        for col in range(input_axes_grid.shape[1]):
            ax = input_axes_grid[bottom_row, col]
            if ax is not None:
                ax.set_xlabel(xlabel)