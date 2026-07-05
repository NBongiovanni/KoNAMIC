from __future__ import annotations

from pathlib import Path
from typing import Sequence

from KoNAMIC.core.systems import SystemSpec
from KoNAMIC.pipelines.experiment_comparison import (
    TrajectoryComparisonResult,
    render_comparison_results,
)
from KoNAMIC.viz import (
    InputComparisonRenderConfig,
    StateComparisonRenderConfig,
)

from .base_visualizer import BaseOpenLoopVisualizer


class OpenLoopMultiVisualizer(BaseOpenLoopVisualizer):
    def __init__(
        self,
        task: str,
        dt: float,
        system_spec: SystemSpec,
        only_position: bool,
        plot_dir: Path,
        names: Sequence[str],
        colors: Sequence[str],
        num_columns_states: int,
        num_columns_inputs: int,
        filename: str,
    ) -> None:
        super().__init__(
            system_dim=system_spec.system_dim,
            dt=dt,
            only_position=only_position,
            num_columns_states=num_columns_states,
            num_columns_inputs=num_columns_inputs,
        )

        self.task = task
        self.dt=dt
        self.system_spec = system_spec
        self.plot_dir = Path(plot_dir)
        self.names = list(names)
        self.colors = list(colors)
        self.filename = filename

        self.x_dim = system_spec.x_dim
        self.u_dim = system_spec.u_dim
        self.x_ref_dim = system_spec.x_ref_dim_open_loop

        self._results_list: list[TrajectoryComparisonResult] | None = None

    def visualize(
        self,
        results_list: Sequence[TrajectoryComparisonResult],
    ) -> None:
        if len(results_list) != len(self.names):
            raise ValueError(
                "results_list must contain one result per compared experiment: "
                f"got {len(results_list)} results for {len(self.names)} names."
            )

        self._results_list = list(results_list)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.run_with_rc_context()

    def _plot(self) -> None:
        assert self._results_list is not None

        x_labels = self.system_spec.get_x_labels(self.only_position)
        u_labels = self.system_spec.get_u_labels()
        render_comparison_results(
            layout_owner=self,
            plot_dir=self.plot_dir,
            filename=self.filename,
            results=self._results_list,
            names=self.names,
            colors=self.colors,
            x_labels=x_labels,
            u_labels=u_labels,
            x_ref_dim=self.x_ref_dim,
            state_config=StateComparisonRenderConfig(
                gt_label="True trajectory",
            ),
            input_config=InputComparisonRenderConfig(
                system_dim=self.system_spec.system_dim,
                grouped_ylabel=r"$\tau$ [N$\cdot$m]",
            ),
        )
