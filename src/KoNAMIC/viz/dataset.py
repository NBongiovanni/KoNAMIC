from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from KoNAMIC.core.systems import SystemSpec
from KoNAMIC.viz.base_visualizer import BaseStateInputVisualizer
from KoNAMIC.viz.input_plot_groups import build_input_plot_groups
from KoNAMIC.viz.single_plot_data import SingleStateInputPlotData
from KoNAMIC.viz.state_input_comparison_renderer import render_single_state_input
from KoNAMIC.viz.state_series import StatePlotSeries
from KoNAMIC.viz.style import COLORS, GT_COLOR


class _DatasetStateInputLayoutOwner(BaseStateInputVisualizer):
    WIDTH_FIGURES = 14
    HEIGHT_FIGURES = 1.8

    def _plot(self) -> None:
        raise NotImplementedError


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

    input_groups = build_input_plot_groups(system, group_inputs=True)
    plot_data = SingleStateInputPlotData(
        time_x=time,
        state_series=[
            StatePlotSeries(
                label="State trajectory",
                values=x_plot,
                color=COLORS[0],
                linestyle="-",
            ),
            StatePlotSeries(
                label="Reference",
                values=x_ref_plot,
                color=GT_COLOR,
                linestyle=":",
            ),
        ],
        x_labels=x_labels,
        time_u=time,
        u=u,
        input_groups=input_groups,
    )

    render_single_state_input(
        layout_owner=_DatasetStateInputLayoutOwner(
            only_position=only_positions,
            num_columns_states=num_columns_states,
            num_columns_inputs=num_columns_inputs,
        ),
        plot_dir=save_dir,
        filename=f"{prefix}_traj_{traj_idx:02d}.png",
        plot_data=plot_data,
        state_grid=True,
        align_ylabels=True,
        state_title="States",
        input_title="Inputs",
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


def _validate_num_columns(value: int, *, name: str) -> None:
    if value not in (1, 2):
        raise ValueError(f"{name} must be either 1 or 2, got {value}.")
