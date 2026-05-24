from __future__ import annotations

from typing import Any, Callable, Optional
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from KoNAMIC.viz.primitives_single import plot_x, plot_u
from KoNAMIC.core.drone import get_u_labels, get_x_labels
from KoNAMIC.viz.axes_layout import get_shared_ylim_groups_state, apply_shared_ylims
from KoNAMIC.viz.style import save_figure

from .base_visualizer import BaseOpenLoopVisualizer

GetArray = Callable[[Any], np.ndarray]
GetOptArray = Callable[[Any], Optional[np.ndarray]]


@dataclass(frozen=True)
class SinglePlotExtractors:
    get_x_gt: GetArray
    get_x_pred: GetOptArray
    get_u: GetArray


class OpenLoopSingleVisualizer(BaseOpenLoopVisualizer):
    def __init__(
        self,
        drone_dim: int,
        dt: float,
        num_columns_states: int,
        num_columns_inputs: int,
        only_position: bool,
        path: Path,
        extractors: SinglePlotExtractors,
    ) -> None:
        super().__init__(
            drone_dim=drone_dim,
            dt=dt,
            only_position=only_position,
            num_columns_states=num_columns_states,
            num_columns_inputs=num_columns_inputs,
        )
        self.path = Path(path)
        self.extractors = extractors
        self.output: Any = None

    def pipeline(self, output: Any) -> None:
        self.output = output
        self.path.mkdir(parents=True, exist_ok=True)
        self.run_with_rc_context()

    def _plot(self) -> None:
        x_gt, x_pred, time_x, x_labels = self._prepare_state_plot_data()
        u, time_u, u_labels = self._prepare_input_plot_data()

        x_gt_disp, x_pred_disp, x_labels_disp = self._get_displayed_state_data(
            x_gt=x_gt,
            x_pred=x_pred,
            labels=x_labels,
        )

        n_states = x_gt_disp.shape[1]
        n_inputs = 2

        fig, state_axes_grid, state_axes_used, input_axes_grid, input_axes_used = (
            self._make_states_inputs_layout(
                n_states=n_states,
                n_inputs=n_inputs,
            )
        )

        self._plot_states_on_axes(
            axes_used=state_axes_used,
            time=time_x,
            x_gt=x_gt_disp,
            x_pred=x_pred_disp,
            labels=x_labels_disp,
        )

        self._plot_inputs_on_axes(
            axes_used=input_axes_used,
            time=time_u,
            u=u,
            labels=u_labels,
        )

        self._hide_inner_xlabels_block(
            axes_grid=state_axes_grid,
            n_cols=self.num_columns_states,
        )
        self._hide_inner_xlabels_block(
            axes_grid=input_axes_grid,
            n_cols=self.num_columns_inputs,
        )
        self._keep_only_bottom_xlabel(input_axes_grid)

        groups = get_shared_ylim_groups_state(self.drone_dim, self.only_position)
        apply_shared_ylims(
            state_axes_grid,
            x_gt_disp,
            x_pred_disp,
            groups_1based=groups,
            pad_frac=0.05,
        )

        save_figure(fig, self.path, "states_and_inputs.pdf")

    def _prepare_state_plot_data(
        self,
    ) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray, list[str]]:
        x_gt = self.extractors.get_x_gt(self.output)
        x_pred = self.extractors.get_x_pred(self.output)
        labels = get_x_labels(self.drone_dim, self.only_position)

        num_steps = x_gt.shape[0]
        time = np.arange(num_steps) * self.dt
        return x_gt, x_pred, time, labels

    def _prepare_input_plot_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        u = self.extractors.get_u(self.output)
        labels = get_u_labels(self.drone_dim)
        time = np.arange(u.shape[0]) * self.dt
        return u, time, labels

    def _get_displayed_state_data(
        self,
        x_gt: np.ndarray,
        x_pred: Optional[np.ndarray],
        labels: list[str],
    ) -> tuple[np.ndarray, Optional[np.ndarray], list[str]]:
        if not self.only_position:
            return x_gt, x_pred, labels

        x_dim_displayed = x_gt.shape[1] // 2
        x_gt_disp = x_gt[:, :x_dim_displayed]
        x_pred_disp = None if x_pred is None else x_pred[:, :x_dim_displayed]
        labels_disp = labels[:x_dim_displayed]
        return x_gt_disp, x_pred_disp, labels_disp

    def _plot_states_on_axes(
        self,
        axes_used: list,
        time: np.ndarray,
        x_gt: np.ndarray,
        x_pred: Optional[np.ndarray],
        labels: list[str],
    ) -> None:
        first = True
        for i, ax in enumerate(axes_used):
            plot_x(
                [ax],
                time,
                [labels[i]],
                None if x_pred is None else x_pred[:, i:i + 1],
                x_gt[:, i:i + 1],
                show_legend=first,
            )
            first = False

    def _plot_inputs_on_axes(
        self,
        axes_used: list,
        time: np.ndarray,
        u: np.ndarray,
        labels: list[str],
    ) -> None:
        plot_u(axes_used, time, u, labels, self.drone_dim)