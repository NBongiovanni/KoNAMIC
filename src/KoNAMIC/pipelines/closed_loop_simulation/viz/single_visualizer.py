from pathlib import Path

import numpy as np

from KoNAMIC.core.systems import SystemSpec
from KoNAMIC.core.simulation import ClosedLoopTrajectory
from KoNAMIC.viz import (
    SingleStateInputPlotData,
    StatePlotSeries,
    build_input_plot_groups,
    render_single_state_input,
)
from KoNAMIC.viz.style import COLORS, GT_COLOR

from .base_visualizer import BaseClosedLoopVisualizer


class ClosedLoopSingleVisualizer(BaseClosedLoopVisualizer):
    def __init__(
        self,
        system_spec: SystemSpec,
        results: ClosedLoopTrajectory,
        run_dir: Path,
        only_positions: bool,
        num_columns_states: int,
        num_columns_inputs: int,
    ):
        super().__init__(
            system_spec=system_spec,
            only_position=only_positions,
            num_columns_states=num_columns_states,
            num_columns_inputs=num_columns_inputs,
        )
        self.run_dir = run_dir
        self.plot_dir = run_dir
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.results = results
        self.only_positions = only_positions
        self.system_spec = system_spec
        self.system_dim = system_spec.system_dim

        self.angles_indexes = self.system_spec.angle_indexes
        self.x_dim = system_spec.x_dim
        self.u_dim = system_spec.u_dim
        self.x_ref_dim = system_spec.x_ref_dim_closed_loop

    def visualize(self) -> None:
        self.run_with_rc_context()

    def _plot(self) -> None:
        plot_data = self._prepare_plot_data()
        render_single_state_input(
            layout_owner=self,
            plot_data=plot_data,
            plot_dir=self.plot_dir,
            filename="states_and_inputs.pdf",
            state_grid=True,
        )

    def _prepare_plot_data(self) -> SingleStateInputPlotData:
        time = self.results.time
        x, x_ref, x_labels, x_dim_displayed = self._get_displayed_state_data()
        x, x_ref, x_labels = self._convert_angles_for_display(x, x_ref, x_labels)

        u = self.results.inputs_data.u_physical
        input_groups = build_input_plot_groups(self.system_spec, group_inputs=True)

        return SingleStateInputPlotData(
            time_x=time,
            state_series=[
                StatePlotSeries(
                    label="Closed-loop",
                    values=x,
                    color=COLORS[0],
                    linestyle="-",
                ),
                *(
                    []
                    if x_ref is None
                    else [
                        StatePlotSeries(
                            label="Reference",
                            values=x_ref,
                            color=GT_COLOR,
                            linestyle="--",
                        )
                    ]
                ),
            ],
            x_labels=x_labels[:x_dim_displayed],
            time_u=time,
            u=u,
            input_groups=input_groups,
        )

    def _get_displayed_state_data(self):
        x = self.results.x_data.traj
        x_ref = self.results.x_data.ref_traj
        labels = self.system_spec.get_x_labels(self.only_positions)

        if self.only_positions:
            x_dim_displayed = x.shape[1] // 2
            x = x[:, :x_dim_displayed]
            if x_ref is not None:
                x_ref = x_ref[:, :x_dim_displayed]
            labels = labels[:x_dim_displayed]
        else:
            x_dim_displayed = x.shape[1]

        return x, x_ref, labels, x_dim_displayed

    def _convert_angles_for_display(
            self,
            x: np.ndarray,
            x_ref: np.ndarray | None,
            labels: list[str],
    ) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
        """
        Convertit les angles d'état de rad vers deg pour l'affichage uniquement.
        Les tableaux d'entrée ne sont pas modifiés in-place.
        """
        x_disp = np.asarray(x, dtype=float).copy()
        x_ref_disp = None if x_ref is None else np.asarray(x_ref, dtype=float).copy()
        labels_disp = list(labels)

        # Indices globaux des angles dans l'état complet
        angle_indices_global = list(self.angles_indexes)

        # Si only_positions=True, les angles ne sont normalement pas affichés
        # car seuls les premiers états sont conservés.
        for idx in angle_indices_global:
            if idx < x_disp.shape[1]:
                x_disp[:, idx] = np.rad2deg(x_disp[:, idx])

                if x_ref_disp is not None and idx < x_ref_disp.shape[1]:
                    x_ref_disp[:, idx] = np.rad2deg(x_ref_disp[:, idx])

                # Remplace [rad] par [°] si besoin
                if idx < len(labels_disp):
                    labels_disp[idx] = (
                        labels_disp[idx]
                        .replace("[rad]", "[°]")
                        .replace("(rad)", "(°)")
                    )

        return x_disp, x_ref_disp, labels_disp
