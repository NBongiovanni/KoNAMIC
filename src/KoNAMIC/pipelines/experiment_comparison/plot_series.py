from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .trajectories import TrajectoryComparisonResult


StateOverlaySeries = tuple[np.ndarray, np.ndarray, np.ndarray, str]
InputOverlaySeries = tuple[np.ndarray, np.ndarray, str]


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
) -> list[StateOverlaySeries]:
    if len(results) != len(names):
        raise ValueError(
            "results and names must have the same length: "
            f"got {len(results)} results and {len(names)} names."
        )

    return [
        (result.time, result.reference_state, result.compared_state, name)
        for name, result in zip(names, results)
    ]


def build_input_overlay_series(
    results: Sequence[TrajectoryComparisonResult],
    names: Sequence[str],
) -> list[InputOverlaySeries]:
    if len(results) != len(names):
        raise ValueError(
            "results and names must have the same length: "
            f"got {len(results)} results and {len(names)} names."
        )

    return [
        (result.input_time, result.inputs, name)
        for name, result in zip(names, results)
    ]


def clip_state_overlay_series(
    series: Sequence[StateOverlaySeries],
    x_dim_displayed: int,
) -> list[StateOverlaySeries]:
    clipped: list[StateOverlaySeries] = []

    for time, reference_state, compared_state, name in series:
        clipped.append(
            (
                time,
                reference_state[:, :x_dim_displayed],
                compared_state[:, :x_dim_displayed],
                name,
            )
        )

    return clipped


def prepare_overlay_plot_data(
    *,
    results: Sequence[TrajectoryComparisonResult],
    names: Sequence[str],
    x_labels: Sequence[str],
    x_ref_dim: int,
    n_inputs: int = 2,
) -> OverlayPlotData:
    state_series = build_state_overlay_series(results, names)
    input_series = build_input_overlay_series(results, names)

    x_dim_displayed = len(x_labels)
    if state_series[0][1].shape[1] != x_dim_displayed:
        state_series = clip_state_overlay_series(state_series, x_dim_displayed)

    return OverlayPlotData(
        state_series=state_series,
        input_series=input_series,
        x_dim_displayed=x_dim_displayed,
        n_ref_displayed=min(x_ref_dim, x_dim_displayed),
        n_states=state_series[0][1].shape[1],
        n_inputs=n_inputs,
    )
