from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from KoNAMIC.viz.base_visualizer import StateInputFigureLayout
from KoNAMIC.viz.primitives_multi import (
    InputOverlayLike,
    StateOverlayLike,
    plot_state_multi,
    plot_u_multi,
)
from KoNAMIC.viz.primitives_single import plot_state_series, plot_u
from KoNAMIC.viz.single_plot_data import SingleStateInputPlotData


@dataclass(frozen=True)
class StateComparisonRenderConfig:
    gt_label: str
    metric: str = "NRMSE_fit"
    show_metric: bool = True


@dataclass(frozen=True)
class InputComparisonRenderConfig:
    system_dim: int
    grouped_ylabel: str = r"Moments [N.m]"
    group_legend_labels: Sequence[str] = (r"$\tau_1$", r"$\tau_2$", r"$\tau_3$")


class StateInputComparisonPlotDataLike(Protocol):
    state_series: Sequence[StateOverlayLike]
    input_series: Sequence[InputOverlayLike]
    n_ref_displayed: int
    n_states: int
    n_inputs: int


def render_single_state_input(
    *,
    layout_owner: Any,
    plot_dir: Path,
    filename: str,
    plot_data: SingleStateInputPlotData,
    state_grid: bool = False,
    align_ylabels: bool = False,
    state_title: str | None = None,
    input_title: str | None = None,
) -> None:
    StateInputComparisonRenderer(layout_owner=layout_owner).render_single(
        plot_dir=plot_dir,
        filename=filename,
        plot_data=plot_data,
        state_grid=state_grid,
        align_ylabels=align_ylabels,
        state_title=state_title,
        input_title=input_title,
    )


class StateInputComparisonRenderer:
    def __init__(self, layout_owner: Any) -> None:
        self.layout_owner = layout_owner

    def render(
        self,
        *,
        plot_dir: Path,
        filename: str,
        plot_data: StateInputComparisonPlotDataLike,
        x_labels: Sequence[str],
        u_labels: Sequence[str],
        state_config: StateComparisonRenderConfig,
        input_config: InputComparisonRenderConfig,
    ) -> None:
        self.layout_owner._render_states_inputs_figure(
            n_states=plot_data.n_states,
            n_inputs=plot_data.n_inputs,
            plot_dir=plot_dir,
            filename=filename,
            render_content=lambda layout: self._render_content(
                layout=layout,
                plot_data=plot_data,
                x_labels=x_labels,
                u_labels=u_labels,
                state_config=state_config,
                input_config=input_config,
            ),
            align_ylabels=True,
        )

    def render_single(
        self,
        *,
        plot_dir: Path,
        filename: str,
        plot_data: SingleStateInputPlotData,
        state_grid: bool = False,
        align_ylabels: bool = False,
        state_title: str | None = None,
        input_title: str | None = None,
    ) -> None:
        self.layout_owner._render_states_inputs_figure(
            n_states=plot_data.n_states,
            n_inputs=plot_data.n_inputs,
            plot_dir=plot_dir,
            filename=filename,
            render_content=lambda layout: self._render_single_content(
                layout=layout,
                plot_data=plot_data,
                state_grid=state_grid,
                state_title=state_title,
                input_title=input_title,
            ),
            align_ylabels=align_ylabels,
        )

    @staticmethod
    def _render_content(
        *,
        layout: StateInputFigureLayout,
        plot_data: StateInputComparisonPlotDataLike,
        x_labels: Sequence[str],
        u_labels: Sequence[str],
        state_config: StateComparisonRenderConfig,
        input_config: InputComparisonRenderConfig,
    ) -> None:
        plot_state_multi(
            axes=layout.state_axes_used,
            x_dim=len(x_labels),
            runs=plot_data.state_series,
            labels=x_labels,
            ref_dim=plot_data.n_ref_displayed,
            gt_label=state_config.gt_label,
            metric=state_config.metric,
            show_metric=state_config.show_metric,
        )
        plot_u_multi(
            system_dim=input_config.system_dim,
            axes=layout.input_axes_used,
            runs_u=plot_data.input_series,
            u_labels=u_labels,
            grouped_ylabel=input_config.grouped_ylabel,
            group_legend_labels=input_config.group_legend_labels,
        )

    @staticmethod
    def _render_single_content(
        *,
        layout: StateInputFigureLayout,
        plot_data: SingleStateInputPlotData,
        state_grid: bool,
        state_title: str | None,
        input_title: str | None,
    ) -> None:
        plot_state_series(
            axes=layout.state_axes_used,
            time=plot_data.time_x,
            state_series=plot_data.state_series,
            labels=plot_data.x_labels,
            grid=state_grid,
        )
        plot_u(
            axes=layout.input_axes_used,
            time=plot_data.time_u,
            u_traj=plot_data.u,
            input_groups=plot_data.input_groups,
        )
        if state_title is not None and layout.state_axes_used:
            layout.state_axes_used[0].set_title(state_title)
        if input_title is not None and layout.input_axes_used:
            layout.input_axes_used[0].set_title(input_title)
