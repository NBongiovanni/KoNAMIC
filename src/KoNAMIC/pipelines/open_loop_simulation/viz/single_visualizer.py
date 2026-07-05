from __future__ import annotations

from typing import Any, Callable, Optional
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from KoNAMIC.core.systems import SystemSpec
from KoNAMIC.viz import (
    SingleStateInputPlotData,
    StatePlotSeries,
    build_input_plot_groups,
    render_single_state_input,
)
from KoNAMIC.viz.primitives_single import InputPlotGroup
from KoNAMIC.viz.style import COLORS, GT_COLOR

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
        system_spec: SystemSpec,
        dt: float,
        num_columns_states: int,
        num_columns_inputs: int,
        only_position: bool,
        path: Path,
        extractors: SinglePlotExtractors,
    ) -> None:
        super().__init__(
            system_dim=system_spec.system_dim,
            dt=dt,
            only_position=only_position,
            num_columns_states=num_columns_states,
            num_columns_inputs=num_columns_inputs,
        )
        self.system_spec = system_spec
        self.path = Path(path)
        self.extractors = extractors
        self.output: Any = None

    def pipeline(self, output: Any) -> None:
        self.output = output
        self.path.mkdir(parents=True, exist_ok=True)
        self.run_with_rc_context()

    def _plot(self) -> None:
        plot_data = self._prepare_plot_data()
        render_single_state_input(
            layout_owner=self,
            plot_data=plot_data,
            plot_dir=self.path,
            filename="states_and_inputs.pdf",
        )

    def _prepare_plot_data(self) -> SingleStateInputPlotData:
        x_gt, x_pred, time_x, x_labels = self._prepare_state_plot_data()
        u, time_u, input_groups = self._prepare_input_plot_data()

        x_gt_disp, x_pred_disp, x_labels_disp = self._get_displayed_state_data(
            x_gt=x_gt,
            x_pred=x_pred,
            labels=x_labels,
        )

        return SingleStateInputPlotData(
            time_x=time_x,
            state_series=[
                *(
                    []
                    if x_pred_disp is None
                    else [
                        StatePlotSeries(
                            label="Prediction",
                            values=x_pred_disp,
                            color=COLORS[0],
                            linestyle="-",
                        )
                    ]
                ),
                StatePlotSeries(
                    label="Ground truth",
                    values=x_gt_disp,
                    color=GT_COLOR,
                    linestyle="--",
                ),
            ],
            x_labels=x_labels_disp,
            time_u=time_u,
            u=u,
            input_groups=input_groups,
        )

    def _prepare_state_plot_data(
        self,
    ) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray, list[str]]:
        x_gt = self.extractors.get_x_gt(self.output)
        x_pred = self.extractors.get_x_pred(self.output)
        labels = self.system_spec.get_x_labels(self.only_position)

        num_steps = x_gt.shape[0]
        time = np.arange(num_steps) * self.dt
        return x_gt, x_pred, time, labels

    def _prepare_input_plot_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, list[InputPlotGroup]]:
        u = self.extractors.get_u(self.output)
        time = np.arange(u.shape[0] + 1) * self.dt
        return u, time, build_input_plot_groups(self.system_spec, group_inputs=True)

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
