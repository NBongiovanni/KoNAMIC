from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from KoNAMIC.core.systems import SystemSpec
from KoNAMIC.viz.primitives_single import InputPlotGroup, plot_x, plot_u
from KoNAMIC.viz.style import save_figure


def plot_dataset_trajectories(
    *,
    dataset,
    system: SystemSpec,
    traj_idx: int,
    save_dir: Path,
    prefix: str,
    only_positions: bool = False,
    num_columns_states: int = 1,
    num_columns_inputs: int = 1,
) -> None:
    _validate_num_columns(num_columns_states, name="num_columns_states")
    _validate_num_columns(num_columns_inputs, name="num_columns_inputs")

    time = dataset.time
    x = dataset.states[traj_idx]
    u = dataset.inputs[traj_idx]
    x_ref = dataset.states_ref[traj_idx]

    # --------------------------------------------------
    # Convert angles from rad to deg for plotting
    # --------------------------------------------------
    x = system.convert_available_angles_to_deg(x)
    x_ref = system.convert_available_angles_to_deg(x_ref)

    # --------------------------------------------------
    # Select state dimensions to plot
    # --------------------------------------------------
    x_plot, x_ref_plot, x_labels = prepare_state_plot_data(
        system=system,
        x=x,
        x_ref=x_ref,
        only_positions=only_positions,
    )

    n_states = x_plot.shape[1]

    # --------------------------------------------------
    # Input groups
    # --------------------------------------------------
    input_groups = build_input_plot_groups(system)
    n_inputs = len(input_groups)

    # --------------------------------------------------
    # Layout: states block above, inputs block below
    # --------------------------------------------------
    fig, state_axes_grid, state_axes_used, input_axes_grid, input_axes_used = (
        make_states_inputs_layout(
            n_states=n_states,
            n_inputs=n_inputs,
            num_columns_states=num_columns_states,
            num_columns_inputs=num_columns_inputs,
        )
    )

    # --------------------------------------------------
    # States
    # --------------------------------------------------
    plot_x(
        axes=state_axes_used,
        time=time,
        labels=x_labels,
        x_main=x_plot,
        x_other=x_ref_plot,
        show_legend=True,
        label_main="State trajectory",
        label_other="Reference",
        main_linestyle="-",
        other_linestyle=":",
    )

    # --------------------------------------------------
    # Inputs
    # --------------------------------------------------
    plot_u(
        axes=input_axes_used,
        time=time,
        u_traj=u,
        input_groups=input_groups,
    )

    # --------------------------------------------------
    # Axis labels cleanup
    # --------------------------------------------------
    hide_inner_xlabels_block(
        axes_grid=state_axes_grid,
        n_cols=num_columns_states,
    )
    hide_inner_xlabels_block(
        axes_grid=input_axes_grid,
        n_cols=num_columns_inputs,
    )
    keep_only_bottom_xlabel(input_axes_grid)

    # --------------------------------------------------
    # Optional titles
    # --------------------------------------------------
    if state_axes_used:
        state_axes_used[0].set_title("States")

    if input_axes_used:
        input_axes_used[0].set_title("Inputs")

    fig.align_ylabels(state_axes_used + input_axes_used)

    save_figure(
        fig,
        save_dir,
        f"{prefix}_traj_{traj_idx:02d}.png",
    )


def plot_dataset_diagnostics(
    *,
    dataset,
    system: SystemSpec,
    save_dir: Path,
    split_name: str,
    traj_indices: Sequence[int] = (0, 1, 2, 4),
    only_positions: bool = False,
    num_columns_states: int = 1,
    num_columns_inputs: int = 1,
) -> None:
    save_dir = Path(save_dir)

    for traj_idx in traj_indices:
        if traj_idx >= dataset.states.shape[0]:
            continue

        plot_dataset_trajectories(
            dataset=dataset,
            system=system,
            traj_idx=traj_idx,
            save_dir=save_dir,
            prefix=split_name,
            only_positions=only_positions,
            num_columns_states=num_columns_states,
            num_columns_inputs=num_columns_inputs,
        )

def prepare_state_plot_data(
    *,
    system: SystemSpec,
    x: np.ndarray,
    x_ref: np.ndarray,
    only_positions: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    full_labels = system.get_x_labels(only_positions=False)

    if only_positions:
        indices = get_main_state_indices(system)
    else:
        indices = list(range(x.shape[1]))

    x_plot = x[:, indices]
    x_ref_plot = x_ref[:, indices]
    labels = [full_labels[i] for i in indices]

    if x_plot.shape[1] != len(labels):
        raise ValueError(
            f"Inconsistent state plot dimensions for system {system.system_name!r}: "
            f"x_plot has {x_plot.shape[1]} columns, but got {len(labels)} labels."
        )

    return x_plot, x_ref_plot, labels


def get_main_state_indices(system: SystemSpec) -> list[int]:
    if system.system_name == "quadrotor_2d":
        return [0, 1]

    if system.system_name == "quadrotor_3d":
        return [0, 1, 2]

    if system.system_name == "cartpole":
        return [0, 1]

    raise ValueError(
        f"Unsupported system_name for main state plot indices: {system.system_name!r}"
    )


def build_input_plot_groups(system: SystemSpec) -> list[InputPlotGroup]:
    if system.system_name == "quadrotor_2d":
        return [
            InputPlotGroup(indices=(0,), ylabel="F [N]"),
            InputPlotGroup(indices=(1,), ylabel=r"$\tau$ [N.m]"),
        ]

    if system.system_name == "quadrotor_3d":
        return [
            InputPlotGroup(indices=(0,), ylabel="F [N]"),
            InputPlotGroup(
                indices=(1, 2, 3),
                ylabel=r"$\tau$ [N.m]",
                legend_labels=(r"$\tau_x$", r"$\tau_y$", r"$\tau_z$"),
            ),
        ]

    if system.system_name == "cartpole":
        return [
            InputPlotGroup(indices=(0,), ylabel="Force [N]"),
        ]

    raise ValueError(
        f"Unsupported system_name for input plot groups: {system.system_name!r}"
    )


def make_states_inputs_layout(
    *,
    n_states: int,
    n_inputs: int,
    num_columns_states: int,
    num_columns_inputs: int,
) -> tuple[
    Figure,
    npt.NDArray[object],
    list[Axes],
    npt.NDArray[object],
    list[Axes],
]:
    n_state_rows, n_state_cols = compute_grid_shape(
        n_plots=n_states,
        n_cols=num_columns_states,
    )
    n_input_rows, n_input_cols = compute_grid_shape(
        n_plots=n_inputs,
        n_cols=num_columns_inputs,
    )

    total_rows = n_state_rows + n_input_rows

    fig = plt.figure(
        figsize=(14, 1.8 * total_rows),
        constrained_layout=True,
    )

    gs = gridspec.GridSpec(total_rows, ncols=2, figure=fig)

    state_axes_grid, state_axes_used = build_block_axes(
        fig=fig,
        gs=gs,
        start_row=0,
        n_plots=n_states,
        n_rows=n_state_rows,
        n_cols=n_state_cols,
    )

    input_axes_grid, input_axes_used = build_block_axes(
        fig=fig,
        gs=gs,
        start_row=n_state_rows,
        n_plots=n_inputs,
        n_rows=n_input_rows,
        n_cols=n_input_cols,
    )

    return fig, state_axes_grid, state_axes_used, input_axes_grid, input_axes_used


def compute_grid_shape(
    *,
    n_plots: int,
    n_cols: int,
) -> tuple[int, int]:
    if n_plots <= 0:
        raise ValueError(f"n_plots must be positive, got {n_plots}.")

    _validate_num_columns(n_cols, name="n_cols")

    n_rows = int(np.ceil(n_plots / n_cols))
    return n_rows, n_cols


def build_block_axes(
    *,
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


def hide_inner_xlabels_block(
    axes_grid: npt.NDArray[object],
    n_cols: int,
) -> None:
    n_rows = axes_grid.shape[0]

    for row in range(n_rows - 1):
        for col in range(n_cols):
            ax = axes_grid[row, col]
            if ax is not None:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)


def keep_only_bottom_xlabel(
    input_axes_grid: npt.NDArray[object],
    xlabel: str = "Time [s]",
) -> None:
    for ax in input_axes_grid.flatten():
        if ax is not None:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)

    bottom_row = input_axes_grid.shape[0] - 1

    for col in range(input_axes_grid.shape[1]):
        ax = input_axes_grid[bottom_row, col]
        if ax is not None:
            ax.set_xlabel(xlabel)
            ax.tick_params(labelbottom=True)


def _validate_num_columns(value: int, *, name: str) -> None:
    if value not in (1, 2):
        raise ValueError(f"{name} must be either 1 or 2, got {value}.")