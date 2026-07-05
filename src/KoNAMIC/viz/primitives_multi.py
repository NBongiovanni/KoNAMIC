from __future__ import annotations
from typing import Iterable, Protocol, Sequence

from matplotlib.axes import Axes
import numpy as np

from .metrics import compute_nrmse_fit, compute_mae_scalar
from .primitives_single import InputPlotGroup, InputPlotSeries, plot_input_series
from .state_series import StatePlotSeries


class StateOverlayLike(Protocol):
    time: np.ndarray
    reference: StatePlotSeries
    compared: Sequence[StatePlotSeries]


class InputOverlayLike(Protocol):
    time: np.ndarray
    values: np.ndarray
    label: str
    color: str | None
    linestyle: str

    def to_plot_series(self, *, color: str | None = None) -> InputPlotSeries:
        ...


ComparedStateRun = tuple[np.ndarray, np.ndarray, str, str | None, str]


def plot_state_multi(
        axes: Sequence[Axes],
        x_dim: int,
        runs: Sequence[StateOverlayLike],
        labels: Sequence[str],
        ref_dim: int,
        gt_label: str,
        colors: Sequence[str] | None = None,
        metric: str = "NRMSE_fit",
        show_metric: bool = True,
) -> None:
    runs = list(runs)
    assert len(runs) > 0
    if metric == "NRMSE_fit":
        metric_fn = compute_nrmse_fit
        unit = "\%"
        metric_name = "fit"
        precision = 0
    elif metric == "MAE":
        metric_fn = compute_mae_scalar
        unit = " m"
        metric_name = "MAE"
        precision = 3
    else:
        raise ValueError(f"Unknown metric: {metric}")

    reference_time, x_ref = _get_shared_reference(runs, ref_dim)
    x_runs = _iter_compared_state_series(runs)

    assert len(axes) >= x_dim, f"Need >= {x_dim} axes, got {len(axes)}"
    assert len(labels) >= x_dim, f"Need >= {x_dim} labels, got {len(labels)}"

    # --- reference once ---
    for i in range(ref_dim):
        if i == 0:
            axes[i].plot(
                reference_time,
                x_ref[:, i],
                color="black",
                linestyle="--",
                label=gt_label,
            )
        else:
            axes[i].plot(
                reference_time,
                x_ref[:, i],
                color="black",
                linestyle="--",
            )

    # --- model_registry ---
    for ridx, (t, x, name, series_color, linestyle) in enumerate(x_runs):
        c = _resolve_series_color(
            series_color=series_color,
            fallback_colors=colors,
            index=ridx,
            label=name,
        )
        T = x.shape[0]
        for i in range(x_dim):
            if show_metric:
                if i < ref_dim:
                    if i==0:
                        value = metric_fn(pred=x[:T, i:i + 1], true=x_ref[:T, i:i + 1])
                        # formatted = format_scientific_latex(value, precision=1)
                        # label = f"{name} -- {metric_name}=${formatted}$ {unit}"
                        label = rf"{name} -- {metric_name}=${value:.{precision}f}$ {unit}"
                        axes[i].plot(t[:T], x[:T, i], color=c, linestyle=linestyle, label=label)
                    else:
                        value = metric_fn(pred=x[:T, i:i + 1], true=x_ref[:T, i:i + 1])
                        # formatted = format_scientific_latex(value, precision=1)
                        label = rf"{metric_name}=${value:.{precision}f}$ {unit}"
                        axes[i].plot(t[:T], x[:T, i], color=c, linestyle=linestyle, label=label)
                else:
                    axes[i].plot(t[:T], x[:T, i], color=c, linestyle=linestyle)
            else:
                if i == 0:
                    label = name
                    axes[i].plot(t[:T], x[:T, i], color=c, linestyle=linestyle, label=label)
                else:
                    axes[i].plot(t[:T], x[:T, i], color=c, linestyle=linestyle)

    for i in range(x_dim):
        axes[i].set_ylabel(labels[i])
        axes[i].grid(True, alpha=0.2)

    for i in range(x_dim):
        if show_metric:
            if i < ref_dim:
                axes[i].legend(loc="best", fontsize=5)
        else:
            if i == 0:
                axes[i].legend(loc="best")

    for i in range(x_dim):
        axes[i].tick_params(labelbottom=False)

    # axes[x_dim - 1].set_xlabel("Time [s]")


def _get_shared_reference(
    runs: Sequence[StateOverlayLike],
    ref_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    reference_time = runs[0].time
    x_ref0 = runs[0].reference.values[:, :ref_dim]

    for run in runs[1:]:
        t = run.time
        x_ref = run.reference.values[:, :ref_dim]
        if (
            t.shape != reference_time.shape
            or not np.allclose(t, reference_time, atol=0.0, rtol=0.0)
        ):
            raise ValueError("Time vectors differ across model_registry; cannot use a single GT.")
        if x_ref.shape != x_ref0.shape or not np.allclose(x_ref, x_ref0):
            raise ValueError("GT differs across model_registry; expected a single shared GT trajectory.")

    return reference_time, x_ref0


def _iter_compared_state_series(
    runs: Sequence[StateOverlayLike],
) -> list[ComparedStateRun]:
    compared_runs: list[ComparedStateRun] = []
    for run in runs:
        if not run.compared:
            raise ValueError("Each state overlay run must contain at least one compared series.")
        compared_runs.extend(
            (
                run.time,
                series.values,
                series.label,
                series.color,
                series.linestyle,
            )
            for series in run.compared
        )

    return compared_runs


def plot_u_multi_2d(
        axes: Sequence[Axes],
        runs_u: Iterable[InputOverlayLike],
        u_labels: Sequence[str],
        *,
        colors: Sequence[str] | None = None,
        show_run_legend: bool = False,
) -> int:
    runs_u = list(runs_u)
    assert len(runs_u) > 0, "runs_u is empty"

    u0 = runs_u[0].values
    u_dim = int(u0.shape[1])
    assert u_dim == 2, f"plot_u_multi_2d expects u_dim=2, got {u_dim}"
    assert len(u_labels) >= 2

    u_axes = axes[:2]
    assert len(u_axes) == 2, "Need 2 axes for u_dim=2"

    input_groups = [
        InputPlotGroup(indices=(0,), ylabel=u_labels[0]),
        InputPlotGroup(indices=(1,), ylabel=u_labels[1]),
    ]
    _plot_multi_input_groups(
        axes=u_axes,
        runs_u=runs_u,
        input_groups=input_groups,
        expected_u_dim=2,
        colors=colors,
        show_series_legend=show_run_legend,
    )

    if not show_run_legend:
        for ax in u_axes:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

    u_axes[-1].set_xlabel("Time [s]")
    return 2


def plot_u_multi_3d(
        axes: Sequence[Axes],
        runs_u: Iterable[InputOverlayLike],
        u_labels: Sequence[str],
        *,
        colors: Sequence[str] | None = None,
        start_idx: int = 0,
        grouped_ylabel: str = r"Moments [N.m]",
        group_legend_labels: Sequence[str] = (r"$\tau_1$", r"$\tau_2$", r"$\tau_3$"),
) -> int:
    runs_u = list(runs_u)
    assert len(runs_u) > 0, "runs_u is empty"

    u0 = runs_u[0].values
    u_dim = int(u0.shape[1])
    assert u_dim == 4, f"plot_u_multi_3d expects u_dim=4, got {u_dim}"
    assert len(u_labels) >= 4

    u_axes = axes[start_idx:start_idx + 2]
    assert len(u_axes) == 2, "Need 2 axes for u_dim=4 (1 + grouped last 3)"

    input_groups = [
        InputPlotGroup(indices=(0,), ylabel=u_labels[0]),
        InputPlotGroup(
            indices=(1, 2, 3),
            ylabel=grouped_ylabel,
            legend_labels=tuple(group_legend_labels),
            linestyles=("-", "--", ":"),
        ),
    ]
    _plot_multi_input_groups(
        axes=u_axes,
        runs_u=runs_u,
        input_groups=input_groups,
        expected_u_dim=4,
        colors=colors,
        show_series_legend=False,
    )

    u_axes[-1].set_xlabel("Time [s]")
    return 2


def plot_u_multi(
        system_dim: int,
        axes: Sequence[Axes],
        runs_u: Iterable[InputOverlayLike],
        u_labels: Sequence[str],
        *,
        colors: Sequence[str] | None = None,
        grouped_ylabel: str = r"Moments [N.m]",
        group_legend_labels: Sequence[str] = (r"$\tau_1$", r"$\tau_2$", r"$\tau_3$"),
) -> int:
    runs_u = list(runs_u)
    assert len(runs_u) > 0
    u_dim = int(runs_u[0].values.shape[1])

    if system_dim == 2:
        return plot_u_multi_2d(
            axes=axes,
            runs_u=runs_u,
            u_labels=u_labels,
            colors=colors,
            show_run_legend=False,
        )
    elif system_dim == 3:
        return plot_u_multi_3d(
            axes=axes,
            runs_u=runs_u,
            u_labels=u_labels,
            colors=colors,
            grouped_ylabel=grouped_ylabel,
            group_legend_labels=group_legend_labels,
        )
    else:
        raise ValueError(f"Unsupported u_dim={u_dim}. Expected 2 or 4.")


def _plot_multi_input_groups(
    *,
    axes: Sequence[Axes],
    runs_u: Sequence[InputOverlayLike],
    input_groups: Sequence[InputPlotGroup],
    expected_u_dim: int,
    colors: Sequence[str] | None,
    show_series_legend: bool,
) -> None:
    for run in runs_u:
        assert run.values.shape[1] == expected_u_dim, (
            f"Expected u_dim={expected_u_dim}, got {run.values.shape[1]}"
        )

    plot_input_series(
        axes=axes,
        input_series=_build_multi_input_series(runs_u, colors=colors),
        input_groups=input_groups,
        show_series_legend=show_series_legend,
    )


def _build_multi_input_series(
    runs_u: Sequence[InputOverlayLike],
    *,
    colors: Sequence[str] | None,
) -> list[InputPlotSeries]:
    return [
        run.to_plot_series(
            color=_resolve_series_color(
                series_color=run.color,
                fallback_colors=colors,
                index=idx,
                label=run.label,
            ),
        )
        for idx, run in enumerate(runs_u)
    ]


def _resolve_series_color(
    *,
    series_color: str | None,
    fallback_colors: Sequence[str] | None,
    index: int,
    label: str,
) -> str:
    if series_color is not None:
        return series_color

    if fallback_colors is not None and index < len(fallback_colors):
        return fallback_colors[index]

    raise ValueError(
        f"No color available for series {label!r}. "
        "Set the series color or provide fallback colors."
    )
