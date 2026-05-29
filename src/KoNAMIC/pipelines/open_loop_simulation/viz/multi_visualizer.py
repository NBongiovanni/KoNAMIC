from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from KoNAMIC.core.drone import get_dimensions
from KoNAMIC.core.drone import get_u_labels, get_x_labels
from KoNAMIC.viz.axes_layout import apply_shared_ylims, get_shared_ylim_groups_state
from KoNAMIC.viz.style import save_figure
from KoNAMIC.viz.primitives_multi import plot_u_multi, plot_state_multi

from .base_visualizer import BaseOpenLoopVisualizer
from .utils.multi_extractors import MultiPlotExtractors, PreparedRunU, PreparedRunX


class OpenLoopMultiVisualizer(BaseOpenLoopVisualizer):
    def __init__(
        self,
        task: str,
        dt: float,
        drone_dim: int,
        only_position: bool,
        plot_dir: Path,
        names: Sequence[str],
        colors: Sequence[str],
        num_columns_states: int,
        num_columns_inputs: int,
        filename: str,
        extractors: MultiPlotExtractors,
    ) -> None:
        super().__init__(
            drone_dim=drone_dim,
            dt=dt,
            only_position=only_position,
            num_columns_states=num_columns_states,
            num_columns_inputs=num_columns_inputs,
        )

        self.task = task
        self.dt=dt
        self.plot_dir = Path(plot_dir)
        self.names = list(names)
        self.colors = list(colors)
        self.filename = filename
        self.extractors = extractors

        self.x_dim, self.u_dim, self.x_ref_dim = get_dimensions(
            drone_dim,
            task=self.task,
        )

        self.results_list: Optional[list[Any]] = None

    def pipeline(self) -> None:
        assert self.results_list is not None
        assert len(self.results_list) == len(self.names)

        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.run_with_rc_context()

    def _plot(self) -> None:
        runs_x = self._iter_runs_x()
        runs_u = self._iter_runs_u()

        x_labels = get_x_labels(self.drone_dim, self.only_position)
        u_labels = get_u_labels(self.drone_dim)

        x_dim_disp = len(x_labels)
        _, x_gt0, x_pred0, _ = runs_x[0]

        if x_gt0.shape[1] != x_dim_disp:
            runs_x = [
                (t, x_gt[:, :x_dim_disp], x_pred[:, :x_dim_disp], name)
                for (t, x_gt, x_pred, name) in runs_x
            ]

        n_ref_disp = min(self.x_ref_dim, x_dim_disp)

        n_states = runs_x[0][1].shape[1]
        n_inputs = 2

        fig, state_axes_grid, state_axes_used, input_axes_grid, input_axes_used = (
            self._make_states_inputs_layout(
                n_states=n_states,
                n_inputs=n_inputs,
            )
        )

        self._plot_states_on_axes(
            fig=fig,
            state_axes_grid=state_axes_grid,
            state_axes_used=state_axes_used,
            runs_x=runs_x,
            x_labels=x_labels,
            n_ref_disp=n_ref_disp,
        )

        self._plot_inputs_on_axes(
            fig=fig,
            input_axes_used=input_axes_used,
            runs_u=runs_u,
            u_labels=u_labels,
        )

        self._hide_inner_xlabels_block(
            axes_grid=state_axes_grid,
            n_cols=self.num_columns_states,
        )
        self._hide_inner_xlabels_block(
            axes_grid=input_axes_grid,
            n_cols=self.num_columns_inputs,
        )
        self._keep_only_bottom_xlabel(input_axes_grid)

        fig.align_ylabels(state_axes_used + input_axes_used)
        save_figure(fig, self.plot_dir, self.filename)

    def _iter_runs_x(self) -> list[PreparedRunX]:
        assert self.results_list is not None

        runs_x: list[PreparedRunX] = []
        for name, out in zip(self.names, self.results_list):
            t, x_gt, x_pred = self.extractors.get_run_x(out, self.dt)
            runs_x.append((t, x_gt, x_pred, name))
        return runs_x

    def _iter_runs_u(self) -> list[PreparedRunU]:
        assert self.results_list is not None

        runs_u: list[PreparedRunU] = []
        for name, out in zip(self.names, self.results_list):
            u = self.extractors.get_u(out)
            t = np.arange(u.shape[0]) * self.dt
            runs_u.append((t, u, name))
        return runs_u

    def _plot_states_on_axes(
        self,
        fig,
        state_axes_grid,
        state_axes_used,
        runs_x: Sequence[PreparedRunX],
        x_labels: Sequence[str],
        n_ref_disp: int,
    ) -> None:
        plot_state_multi(
            axes=state_axes_used,
            x_dim=len(x_labels),
            runs=runs_x,
            names=self.names,
            labels=x_labels,
            colors=self.colors,
            ref_dim=n_ref_disp,
            gt_label="True trajectory",
        )

        groups = get_shared_ylim_groups_state(self.drone_dim, self.only_position)
        x_gt_all = np.concatenate([r[1] for r in runs_x], axis=0)
        x_pred_all = np.concatenate([r[2] for r in runs_x], axis=0)

        apply_shared_ylims(
            axes=state_axes_grid,
            x_gt=x_gt_all,
            x_pred=x_pred_all,
            groups_1based=groups,
            pad_frac=0.05,
        )

    def _plot_inputs_on_axes(
        self,
        fig,
        input_axes_used,
        runs_u: Sequence[PreparedRunU],
        u_labels: Sequence[str],
    ) -> None:
        plot_u_multi(
            drone_dim=self.drone_dim,
            axes=input_axes_used,
            runs_u=runs_u,
            u_labels=u_labels,
            grouped_ylabel=r"$\tau$ [N$\cdot$m]",
            colors=["gray", "gray", "gray"],
        )