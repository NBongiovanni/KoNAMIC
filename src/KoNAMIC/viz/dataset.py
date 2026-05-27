# src/KoNAMIC/viz/dataset.py

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

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
    # Figure 1: states + references
    # --------------------------------------------------
    fig_x, axes_x = plt.subplots(
        n_x_plot,
        1,
        figsize=(8, 1.7 * n_x_plot),
        sharex=True,
    )

    axes_x = np.atleast_1d(axes_x)

    plot_x_gt(
        axes=axes_x,
        time=time,
        labels=x_labels,
        x_gt=x_plot,
        show_legend=True,
        label_gt="State trajectory",
    )

    # Ajout de la référence sur les composantes disponibles
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

    fig_x.tight_layout()
    save_figure(
        fig_x,
        save_dir,
        f"{prefix}_traj_{traj_idx:03d}_states.png",
    )

    # --------------------------------------------------
    # Figure 2: inputs
    # --------------------------------------------------
    if drone.drone_dim == 2:
        n_u_axes = 2
    elif drone.drone_dim == 3:
        # Votre plot_u_3d regroupe les 3 moments sur un même axe :
        # 1 axe pour T + 1 axe pour les moments.
        n_u_axes = 2
    else:
        raise ValueError(f"Unsupported drone_dim: {drone.drone_dim}")

    fig_u, axes_u = plt.subplots(
        n_u_axes,
        1,
        figsize=(8, 2.0 * n_u_axes),
        sharex=True,
    )

    axes_u = np.atleast_1d(axes_u)

    plot_u(
        axes=axes_u,
        time=time,
        u_traj=u,
        labels=u_labels,
        drone_dim=drone.drone_dim,
    )

    fig_u.tight_layout()
    save_figure(
        fig_u,
        save_dir,
        f"{prefix}_traj_{traj_idx:03d}_inputs.png",
    )


def plot_dataset_diagnostics(
    *,
    dataset,
    drone: DroneSpec,
    save_dir: Path,
    split_name: str,
    traj_indices: Sequence[int] = (0, 1, 2),
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