from pathlib import Path
from typing import Sequence

from KoNAMIC.viz import (
    InputComparisonRenderConfig,
    StateComparisonRenderConfig,
)

from KoNAMIC.core.systems import SystemSpec
from KoNAMIC.pipelines.experiment_comparison import (
    TrajectoryComparisonResult,
    render_comparison_results,
)
from .base_visualizer import BaseClosedLoopVisualizer


class ClosedLoopMultiVisualizer(BaseClosedLoopVisualizer):
    def __init__(
        self,
        system_spec: SystemSpec,
        plot_dir: Path,
        dt: float,
        names: Sequence[str],
        colors: Sequence[str],
        only_position: bool,
        num_columns_states: int,
        num_columns_inputs: int,
        filename: str = "closed_loop_simulation.pdf",
    ) -> None:
        super().__init__(
            system_spec=system_spec,
            only_position=only_position,
            num_columns_states=num_columns_states,
            num_columns_inputs=num_columns_inputs,
        )

        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        self.dt = float(dt)
        self.names = list(names)
        self.colors = list(colors)
        self.filename = filename
        self._results_list: list[TrajectoryComparisonResult] | None = None
        self.system = system_spec

        self.x_dim = system_spec.x_dim
        self.u_dim = system_spec.u_dim
        self.x_ref_dim = system_spec.x_ref_dim_closed_loop

    def visualize(
        self,
        results_list: Sequence[TrajectoryComparisonResult],
    ) -> None:
        if len(results_list) != len(self.names):
            raise ValueError(
                "results_list must contain one result per compared source: "
                f"got {len(results_list)} results for {len(self.names)} names."
            )

        self._results_list = list(results_list)
        self.run_with_rc_context()

    def _plot(self) -> None:
        assert self._results_list is not None

        labels_x = self.system.get_x_labels(self.only_position)
        labels_u = self.system.get_u_labels()
        render_comparison_results(
            layout_owner=self,
            plot_dir=self.plot_dir,
            filename=self.filename,
            results=self._results_list,
            names=self.names,
            colors=self.colors,
            x_labels=labels_x,
            u_labels=labels_u,
            x_ref_dim=self.x_ref_dim,
            state_config=StateComparisonRenderConfig(
                gt_label="Reference",
                metric="MAE",
                show_metric=True,
            ),
            input_config=InputComparisonRenderConfig(
                system_dim=self.system_spec.system_dim,
                grouped_ylabel=r"$\tau$ [N.m]",
                group_legend_labels=[r"$\tau_1$", r"$\tau_2$", r"$\tau_3$"],
            ),
        )
