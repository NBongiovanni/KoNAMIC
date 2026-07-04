from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from matplotlib.axes import Axes
import numpy as np

from KoNAMIC.viz.style import GT_COLOR, COLORS


@dataclass(frozen=True)
class InputPlotGroup:
    indices: tuple[int, ...]
    ylabel: str
    legend_labels: tuple[str, ...] | None = None
    linestyles: tuple[str, ...] | None = None


def plot_x(
    axes: Sequence[Axes],
    time: np.ndarray,
    labels: list[str],
    x_main: np.ndarray,
    x_other: np.ndarray | None = None,
    show_legend: bool = True,
    label_main: str = "Closed-loop",
    label_other: str = "Reference",
    main_linestyle: str = "-",
    other_linestyle: str = "--",
    main_color=COLORS[0],
    other_color=GT_COLOR,
) -> None:
    """
    Plot one or two state trajectories.

    This primitive is intentionally system-agnostic.

    Convention:
      - len(axes) == number of dimensions to plot
      - x_main has shape (T, n_dim)
      - x_other, if provided, has shape (T, n_dim)
      - n_dim == len(axes)

    Typical usages:
      - open-loop:
          x_main = true trajectory
          x_other = predicted trajectory

      - closed-loop:
          x_main = reference or true trajectory
          x_other = controlled trajectory

      - dataset diagnostics:
          x_main = state trajectory
          x_other = reference trajectory
    """
    n_dim = len(axes)

    if x_main.ndim != 2:
        raise ValueError(f"x_main must be 2D, got shape {x_main.shape}.")

    if x_main.shape[1] != n_dim:
        raise ValueError(
            f"x_main.shape[1] must be {n_dim}, got {x_main.shape[1]}."
        )

    if len(labels) != n_dim:
        raise ValueError(f"len(labels) must be {n_dim}, got {len(labels)}.")

    if x_other is not None:
        if x_other.ndim != 2:
            raise ValueError(f"x_other must be 2D, got shape {x_other.shape}.")

        if x_other.shape[1] != n_dim:
            raise ValueError(
                f"x_other.shape[1] must be {n_dim}, got {x_other.shape[1]}."
            )

        if x_other.shape[0] != x_main.shape[0]:
            raise ValueError(
                f"x_main and x_other must have the same number of time steps, "
                f"got {x_main.shape[0]} and {x_other.shape[0]}."
            )

    for i, ax in enumerate(axes):
        ax.plot(
            time[1:],
            x_main[1:, i],
            color=main_color,
            linestyle=main_linestyle,
            label=label_main,
        )

        if x_other is not None:
            ax.plot(
                time[1:],
                x_other[1:, i],
                color=other_color,
                linestyle=other_linestyle,
                label=label_other,
            )

        ax.set_ylabel(labels[i])
        ax.tick_params(labelbottom=False)

    if show_legend:
        axes[0].legend(loc="upper right", fontsize=5)


def plot_x_gt(
    axes: Sequence[Axes],
    time: np.ndarray,
    labels: list[str],
    x_gt: np.ndarray,
    show_legend: bool = True,
    label_gt: str = "True trajectory",
) -> None:
    """
    Plot true states.

    Convention:
      - len(axes) == number of dimensions to plot
      - x_gt has shape (T, n_dim)
      - n_dim == len(axes)
    """
    n_dim = len(axes)

    if x_gt.shape[1] != n_dim:
        raise ValueError(f"x_gt.shape[1] must be {n_dim}, got {x_gt.shape[1]}.")

    if len(labels) != n_dim:
        raise ValueError(f"len(labels) must be {n_dim}, got {len(labels)}.")

    for i, ax in enumerate(axes):
        ax.plot(
            time,
            x_gt[:, i],
            color=GT_COLOR,
            linestyle="--",
            label=label_gt,
        )
        ax.set_ylabel(labels[i])
        ax.tick_params(labelbottom=False)
        ax.grid(True, which="major", axis="both", alpha=0.25)

    if show_legend:
        axes[0].legend(loc="upper right", fontsize=5)


def plot_u(
    axes: Sequence[Axes],
    time: np.ndarray,
    u_traj: np.ndarray,
    input_groups: Sequence[InputPlotGroup],
) -> None:
    """
    Plot control inputs using generic input groups.

    Each InputPlotGroup corresponds to one axis.

    Examples:
      - CartPole:
          [InputPlotGroup(indices=(0,), ylabel="force [N]")]
      - Quadrotor 2D:
          [
              InputPlotGroup(indices=(0,), ylabel="F [N]"),
              InputPlotGroup(indices=(1,), ylabel=r"$\\tau$ [N.m]"),
          ]
      - Quadrotor 3D:
          [
              InputPlotGroup(indices=(0,), ylabel="F [N]"),
              InputPlotGroup(
                  indices=(1, 2, 3),
                  ylabel=r"$\\tau$ [N.m]",
                  legend_labels=(r"$\\tau_x$", r"$\\tau_y$", r"$\\tau_z$"),
              ),
          ]
    """
    if len(axes) != len(input_groups):
        raise ValueError(
            f"Expected one axis per input group: "
            f"len(axes)={len(axes)}, len(input_groups)={len(input_groups)}."
        )

    if u_traj.ndim != 2:
        raise ValueError(f"u_traj must be 2D, got shape {u_traj.shape}.")

    u_dim = u_traj.shape[1]

    for ax, group in zip(axes, input_groups):
        _plot_input_group(
            ax=ax,
            time=time,
            u_traj=u_traj,
            group=group,
            u_dim=u_dim,
        )

    axes[-1].set_xlabel("Time [s]")


def _plot_input_group(
    ax: Axes,
    time: np.ndarray,
    u_traj: np.ndarray,
    group: InputPlotGroup,
    u_dim: int,
) -> None:
    if len(group.indices) == 0:
        raise ValueError("InputPlotGroup.indices must contain at least one index.")

    for idx in group.indices:
        if idx < 0 or idx >= u_dim:
            raise ValueError(
                f"Input index {idx} is out of bounds for u_dim={u_dim}."
            )

    if group.legend_labels is not None and len(group.legend_labels) != len(group.indices):
        raise ValueError(
            f"legend_labels must have length {len(group.indices)}, "
            f"got {len(group.legend_labels)}."
        )

    linestyles = _get_linestyles(group)

    for k, input_idx in enumerate(group.indices):
        label = None
        if group.legend_labels is not None:
            label = group.legend_labels[k]

        ax.plot(
            time[:-1],
            u_traj[:, input_idx],
            color="gray",
            linestyle=linestyles[k],
            label=label,
        )

    ax.set_ylabel(group.ylabel)
    ax.grid(True, which="major", axis="both", alpha=0.25)
    ax.set_axisbelow(True)
    ax.yaxis.get_major_formatter().set_powerlimits((0, 0))

    if group.legend_labels is not None:
        ax.legend(loc="upper right", fontsize=5)


def _get_linestyles(group: InputPlotGroup) -> tuple[str, ...]:
    n_inputs = len(group.indices)

    if group.linestyles is not None:
        if len(group.linestyles) != n_inputs:
            raise ValueError(
                f"linestyles must have length {n_inputs}, "
                f"got {len(group.linestyles)}."
            )
        return group.linestyles

    default_styles = ("-", "--", ":", "-.")

    if n_inputs <= len(default_styles):
        return default_styles[:n_inputs]

    return tuple("-" for _ in range(n_inputs))


def plot_z(
    axes: Sequence[Axes],
    time: np.ndarray,
    z_proj: np.ndarray,
    z_pred: np.ndarray,
) -> None:
    """
    Plot projected and predicted latent variables.
    """
    if z_proj.shape != z_pred.shape:
        raise ValueError(
            f"z_proj and z_pred must have the same shape, "
            f"got {z_proj.shape} and {z_pred.shape}."
        )

    if len(axes) != z_pred.shape[1]:
        raise ValueError(
            f"Expected {z_pred.shape[1]} axes, got {len(axes)}."
        )

    for dim in range(z_pred.shape[1]):
        axes[dim].plot(time, z_proj[:, dim], label="z_proj")
        axes[dim].plot(time, z_pred[:, dim], label="z_pred")
        axes[dim].set_ylabel(rf"$z_{{{dim + 1}}}$")

    axes[-1].set_xlabel("Time [s]")