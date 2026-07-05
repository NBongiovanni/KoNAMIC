from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from KoNAMIC.viz import (
    InputComparisonRenderConfig,
    StateComparisonRenderConfig,
    StateInputComparisonRenderer,
)
from KoNAMIC.viz.primitives_single import InputPlotSeries
from KoNAMIC.viz.state_series import StatePlotSeries

from .trajectories import TrajectoryComparisonResult


@dataclass(frozen=True)
class InputOverlaySeries:
    time: np.ndarray
    values: np.ndarray
    label: str
    color: str | None = None
    linestyle: str = "-"

    def to_plot_series(self, *, color: str | None = None) -> InputPlotSeries:
        resolved_color = color or self.color or "gray"
        return InputPlotSeries(
            time=self.time,
            values=self.values,
            label=self.label,
            color=resolved_color,
            linestyle=self.linestyle,
        )


@dataclass(frozen=True)
class StateOverlaySeries:
    time: np.ndarray
    series: tuple[StatePlotSeries, ...]

    def __post_init__(self) -> None:
        if not self.series:
            raise ValueError("series must contain at least one state series.")

    @property
    def reference(self) -> StatePlotSeries:
        return self.series[0]

    @property
    def compared(self) -> tuple[StatePlotSeries, ...]:
        return self.series[1:]


@dataclass(frozen=True)
class OverlayPlotData:
    state_series: list[StateOverlaySeries]
    input_series: list[InputOverlaySeries]
    x_dim_displayed: int
    n_ref_displayed: int
    n_states: int
    n_inputs: int


def build_state_overlay_series(
    results: Sequence[TrajectoryComparisonResult],
    names: Sequence[str],
    colors: Sequence[str] | None = None,
) -> list[StateOverlaySeries]:
    if len(results) != len(names):
        raise ValueError(
            "results and names must have the same length: "
            f"got {len(results)} results and {len(names)} names."
        )
    if colors is not None and len(colors) != len(results):
        raise ValueError(
            "colors must have the same length as results when provided: "
            f"got {len(colors)} colors and {len(results)} results."
        )

    return [
        StateOverlaySeries(
            time=result.time,
            series=(
                StatePlotSeries(
                    label="Reference",
                    values=result.reference_state,
                    color="black",
                    linestyle="--",
                ),
                StatePlotSeries(
                    label=name,
                    values=result.compared_state,
                    color=None if colors is None else color,
                    linestyle="-",
                ),
            ),
        )
        for name, result, color in zip(
            names,
            results,
            [None] * len(results) if colors is None else colors,
        )
    ]


def build_input_overlay_series(
    results: Sequence[TrajectoryComparisonResult],
    names: Sequence[str],
    colors: Sequence[str] | None = None,
) -> list[InputOverlaySeries]:
    if len(results) != len(names):
        raise ValueError(
            "results and names must have the same length: "
            f"got {len(results)} results and {len(names)} names."
        )
    if colors is not None and len(colors) != len(results):
        raise ValueError(
            "colors must have the same length as results when provided: "
            f"got {len(colors)} colors and {len(results)} results."
        )

    return [
        InputOverlaySeries(
            time=result.input_time,
            values=result.inputs,
            label=name,
            color=None if colors is None else color,
            linestyle="-",
        )
        for name, result, color in zip(
            names,
            results,
            [None] * len(results) if colors is None else colors,
        )
    ]


def clip_state_overlay_series(
    series: Sequence[StateOverlaySeries],
    x_dim_displayed: int,
) -> list[StateOverlaySeries]:
    clipped: list[StateOverlaySeries] = []

    for run in series:
        clipped.append(
            StateOverlaySeries(
                time=run.time,
                series=tuple(
                    StatePlotSeries(
                        label=state_series.label,
                        values=state_series.values[:, :x_dim_displayed],
                        color=state_series.color,
                        linestyle=state_series.linestyle,
                    )
                    for state_series in run.series
                ),
            )
        )

    return clipped


def prepare_overlay_plot_data(
    *,
    results: Sequence[TrajectoryComparisonResult],
    names: Sequence[str],
    colors: Sequence[str] | None = None,
    x_labels: Sequence[str],
    x_ref_dim: int,
    n_inputs: int = 2,
) -> OverlayPlotData:
    state_series = build_state_overlay_series(results, names, colors=colors)
    input_series = build_input_overlay_series(results, names, colors=colors)

    x_dim_displayed = len(x_labels)
    if state_series[0].reference.values.shape[1] != x_dim_displayed:
        state_series = clip_state_overlay_series(state_series, x_dim_displayed)

    return OverlayPlotData(
        state_series=state_series,
        input_series=input_series,
        x_dim_displayed=x_dim_displayed,
        n_ref_displayed=min(x_ref_dim, x_dim_displayed),
        n_states=state_series[0].reference.values.shape[1],
        n_inputs=n_inputs,
    )


def render_comparison_results(
    *,
    layout_owner,
    plot_dir: Path,
    filename: str,
    results: Sequence[TrajectoryComparisonResult],
    names: Sequence[str],
    colors: Sequence[str] | None,
    x_labels: Sequence[str],
    u_labels: Sequence[str],
    x_ref_dim: int,
    state_config: StateComparisonRenderConfig,
    input_config: InputComparisonRenderConfig,
) -> None:
    plot_data = prepare_overlay_plot_data(
        results=results,
        names=names,
        colors=colors,
        x_labels=x_labels,
        x_ref_dim=x_ref_dim,
    )

    StateInputComparisonRenderer(layout_owner=layout_owner).render(
        plot_dir=plot_dir,
        filename=filename,
        plot_data=plot_data,
        x_labels=x_labels,
        u_labels=u_labels,
        state_config=state_config,
        input_config=input_config,
    )
