from typing import Sequence

from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import numpy as np

from KoNAMIC.viz.style import GT_COLOR


def plot_x(
    axes: Sequence[Axes],
    time: np.ndarray,
    labels: list[str],
    x: np.ndarray,
    x_gt: np.ndarray,
    show_legend: bool = True,
    label_x: str = "Predicted trajectory",
    label_gt: str = "True trajectory"
) -> None:
    """
    Trace l'état réel et prédit.
    Convention :
      - len(axes) == nombre de dimensions à tracer
      - x, x_gt ont la forme (T, n_dim) avec n_dim == len(axes)
    """
    n_dim = len(axes)

    assert x.shape[1] == n_dim
    assert x_gt.shape[1] == n_dim
    assert len(labels) == n_dim

    for i, ax in enumerate(axes):
        ax.plot(
            time[1:],
            x_gt[1:, i],
            color=GT_COLOR,
            linestyle = "--",
            label=label_gt
        )
        ax.plot(time[1:], x[1:, i], label=label_x)
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
    label_gt: str = "True trajectory"
) -> None:
    """
    Trace l'état réel et prédit.
    Convention :
      - len(axes) == nombre de dimensions à tracer
      - x, x_gt ont la forme (T, n_dim) avec n_dim == len(axes)
    """
    n_dim = len(axes)

    assert x_gt.shape[1] == n_dim
    assert len(labels) == n_dim

    for i, ax in enumerate(axes):
        ax.plot(
            time,
            x_gt[:, i],
            color=GT_COLOR,
            linestyle = "--",
            label=label_gt
        )
        ax.set_ylabel(labels[i])
        ax.tick_params(labelbottom=False)

    #axes[-1].set_xlabel("Time [s]")
    if show_legend:
        axes[0].legend(loc="upper right", fontsize=5)


def plot_u(
    axes: Sequence[Axes],
    time: np.ndarray,
    u_traj: np.ndarray,
    labels: list,
    drone_dim: int,
    *,
    grouped_ylabel: str = r"Moments [N.m]",
    group_legend_labels: Sequence[str] = (r"$\tau_1$", r"$\tau_2$", r"$\tau_3$"),
) -> None:
    if drone_dim == 2:
        plot_u_2d(
            axes,
            time,
            u_traj,
            labels,
        )
    elif drone_dim == 3:
        plot_u_3d(
            axes,
            time,
            u_traj,
            labels,
            grouped_ylabel=grouped_ylabel,
            group_legend_labels=group_legend_labels,
        )
    else:
        raise ValueError(f"Unsupported drone_dim. Expected 2 or 3, got {drone_dim}.")


def plot_u_2d(
    axes: Sequence[Axes],
    time: np.ndarray,
    u_traj: np.ndarray,
    labels: list
) -> None:
    for dim in range(2):
        axes[dim].plot(time, u_traj[:, dim])
        axes[dim].set_ylabel(labels[dim])
    axes[-1].set_xlabel("Time [s]")


def plot_u_3d(
    axes: Sequence[Axes],
    time: np.ndarray,
    u_traj: np.ndarray,
    labels: Sequence[str],
    *,
    start_idx: int = 0,
    grouped_ylabel: str = r"$\tau$ [N.m]",
    group_legend_labels: Sequence[str] = (r"$\tau_1$", r"$\tau_2$", r"$\tau_3$"),
)-> None:

    u_dim = int(u_traj.shape[1])
    assert u_dim == 4, f"plot_u_3d expects u_dim=4, got {u_dim}"
    assert len(labels) >= 4

    u_axes = axes[start_idx:start_idx + 2]
    assert len(u_axes) == 2, "Need 2 axes for u_dim=4 (1 + grouped last 3)"

    ax0, axM = u_axes
    styles = ["-", "--", ":"]

    # Premier input seul
    ax0.plot(time[:-1], u_traj[:, 0])
    ax0.set_ylabel(labels[0])

    # Trois moments regroupés sur le même axe
    for k, j in enumerate([1, 2, 3]):
        axM.plot(time[:-1], u_traj[:, j],  color="black", linestyle=styles[k])

    axM.set_ylabel(grouped_ylabel)

    handles = [Line2D([0], [0], color="black", linestyle=styles[k]) for k in range(3)]
    axM.legend(handles, list(group_legend_labels), loc="upper right", fontsize=5)
    axM.yaxis.get_major_formatter().set_powerlimits((0, 0))

    for ax in u_axes:
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("Time [s]")


def plot_z(
    axes: Sequence[Axes],
    time: np.ndarray,
    z_proj: np.ndarray,
    z_pred: np.ndarray,
) -> None:
    """
    Trace la variable latente réelle et projetée.
    """
    for dim in range(z_pred.shape[1]):
        axes[dim].plot(time, z_proj[:, dim], label="z_proj")
        axes[dim].plot(time, z_pred[:, dim], label="z_pred")
        axes[dim].set_ylabel(rf"$z_{{{dim+1}}}$")
    axes[-1].set_xlabel("Time [s]")

