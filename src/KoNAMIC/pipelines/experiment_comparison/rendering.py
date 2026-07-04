from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from KoNAMIC.viz import save_figure

from .plot_series import InputOverlaySeries, OverlayPlotData, StateOverlaySeries


PlotStatesCallback = Callable[
    [
        Figure,
        npt.NDArray[object],
        list[Axes],
        Sequence[StateOverlaySeries],
        Sequence[str],
        int,
    ],
    None,
]
PlotInputsCallback = Callable[
    [
        Figure,
        list[Axes],
        Sequence[InputOverlaySeries],
        Sequence[str],
    ],
    None,
]


def render_state_input_overlay(
    *,
    layout_owner: Any,
    plot_dir: Path,
    filename: str,
    plot_data: OverlayPlotData,
    x_labels: Sequence[str],
    u_labels: Sequence[str],
    plot_states: PlotStatesCallback,
    plot_inputs: PlotInputsCallback,
) -> None:
    fig, state_axes_grid, state_axes_used, input_axes_grid, input_axes_used = (
        layout_owner._make_states_inputs_layout(
            n_states=plot_data.n_states,
            n_inputs=plot_data.n_inputs,
        )
    )

    plot_states(
        fig,
        state_axes_grid,
        state_axes_used,
        plot_data.state_series,
        x_labels,
        plot_data.n_ref_displayed,
    )
    plot_inputs(
        fig,
        input_axes_used,
        plot_data.input_series,
        u_labels,
    )

    layout_owner._hide_inner_xlabels_block(
        axes_grid=state_axes_grid,
        n_cols=layout_owner.num_columns_states,
    )
    layout_owner._hide_inner_xlabels_block(
        axes_grid=input_axes_grid,
        n_cols=layout_owner.num_columns_inputs,
    )
    layout_owner._keep_only_bottom_xlabel(input_axes_grid)

    fig.align_ylabels(state_axes_used + input_axes_used)
    save_figure(fig, plot_dir, filename)
