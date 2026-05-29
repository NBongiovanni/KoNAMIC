from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt

from KoNAMIC.core.drone import DroneSpec
from KoNAMIC.viz.primitives_single import plot_x_gt, plot_u
from KoNAMIC.viz.style import save_figure


def plot_dataset_trajectory(
    *,
    dataset,
    drone: DroneSpec,
    traj_idx: int,
    save_dir: Path,
    prefix: str,
    only_positions: bool = False,
) -> None:

    time = dataset.time
    x = dataset.states[traj_idx]
    u = dataset.inputs[traj_idx]
    x_ref = dataset.states_ref[traj_idx]

    # --------------------------------------------------
    # Convert angles from rad to deg for plotting
    # --------------------------------------------------
    x = drone.convert_available_angles_to_deg(x)
    x_ref = drone.convert_available_angles_to_deg(x_ref)

    x_labels = drone.get_x_labels(only_positions=only_positions)
    u_labels = drone.get_u_labels()

    if only_positions:
        n_x_plot = drone.x_dim // 2
        x_plot = x[:, :n_x_plot]
        x_ref_plot = x_ref[:, :n_x_plot]
    else:
        n_x_plot = drone.x_dim
        x_plot = x
        x_ref_plot = x_ref
    # --------------------------------------------------
    # Number of input axes
    # --------------------------------------------------
    if drone.drone_dim == 2:
        n_u_axes = 2
    elif drone.drone_dim == 3:
        # plot_u_3d groups the three moments on the same axis:
        # one axis for the thrust and one axis for the moments.
        n_u_axes = 2
    else:
        raise ValueError(f"Unsupported drone_dim: {drone.drone_dim}")

    # --------------------------------------------------
    # Single figure: states on the left, inputs on the right
    # --------------------------------------------------
    n_rows = max(n_x_plot, n_u_axes)

    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(14, 1.8 * n_rows),
        sharex="col",
        squeeze=False,
    )

    axes_x = axes[:n_x_plot, 0]
    axes_u = axes[:n_u_axes, 1]

    # Left column: states + references
    plot_x_gt(
        axes=axes_x,
        time=time,
        labels=x_labels,
        x_gt=x_plot,
        show_legend=True,
        label_gt="State trajectory",
    )

    ref_dim = min(x_ref_plot.shape[1], n_x_plot)
    for j in range(ref_dim):
        axes_x[j].plot(
            time,
            x_ref_plot[:, j],
            linestyle=":",
            label="Reference" if j == 0 else None,
        )

    axes_x[-1].set_xlabel("Time [s]")
    axes_x[0].legend(loc="best", fontsize=7)

    # Right column: inputs
    plot_u(
        axes=axes_u,
        time=time,
        u_traj=u,
        labels=u_labels,
        drone_dim=drone.drone_dim,
    )

    axes_u[-1].set_xlabel("Time [s]")

    # Hide unused axes, if one column has fewer rows than the other.
    for row in range(n_x_plot, n_rows):
        axes[row, 0].set_visible(False)

    for row in range(n_u_axes, n_rows):
        axes[row, 1].set_visible(False)

    # Optional column titles.
    axes[0, 0].set_title("States")
    axes[0, 1].set_title("Inputs")

    fig.tight_layout()

    save_figure(
        fig,
        save_dir,
        f"{prefix}_traj_{traj_idx:02d}.png",
    )


def plot_dataset_diagnostics(
    *,
    dataset,
    drone: DroneSpec,
    save_dir: Path,
    split_name: str,
    traj_indices: Sequence[int] = (0, 1, 2, 4),
    only_positions: bool = False,
) -> None:
    save_dir = Path(save_dir)

    for traj_idx in traj_indices:
        if traj_idx >= dataset.states.shape[0]:
            continue

        plot_dataset_trajectory(
            dataset=dataset,
            drone=drone,
            traj_idx=traj_idx,
            save_dir=save_dir,
            prefix=split_name,
            only_positions=only_positions,
        )