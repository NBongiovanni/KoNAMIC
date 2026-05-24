from pathlib import Path

import numpy as np

from KoNAMIC.viz.axes_layout import get_shared_ylim_groups_state, apply_shared_ylims
from KoNAMIC.core import drone
from KoNAMIC.core.simulation import ClosedLoopTrajectory
from KoNAMIC.viz import save_figure
from KoNAMIC.viz.primitives_single import plot_x, plot_u

from .base_visualizer import BaseClosedLoopVisualizer


class ClosedLoopSingleVisualizer(BaseClosedLoopVisualizer):
    def __init__(
        self,
        drone_dim: int,
        results: ClosedLoopTrajectory,
        run_dir: Path,
        dt: float,
        nominal_control: bool,
        only_positions: bool,
        num_columns_states: int,
        num_columns_inputs: int,
    ):
        super().__init__(
            drone_dim=drone_dim,
            dt=dt,
            only_position=only_positions,
            num_columns_states=num_columns_states,
            num_columns_inputs=num_columns_inputs,
        )
        self.run_dir = run_dir
        self.plot_dir = run_dir
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.results = results
        self.nominal_control = nominal_control
        self.only_positions = only_positions

        self.angles_indexes = drone.get_angle_indexes(drone_dim)
        self.x_dim, self.u_dim, self.x_ref_dim = drone.get_dimensions(drone_dim)

    def visualize(self) -> None:
        self.run_with_rc_context()

    def _plot(self) -> None:
        time = self.results.time
        x, x_ref, x_labels, x_dim_displayed = self._get_displayed_state_data()
        x, x_ref, x_labels = self._convert_angles_for_display(x, x_ref, x_labels)

        u = self.results.inputs_data.u_physical
        u_labels = drone.get_u_labels(self.drone_dim)

        n_states = x_dim_displayed
        n_inputs = 2

        fig, state_axes_grid, state_axes_used, input_axes_grid, input_axes_used = (
            self._make_states_inputs_layout(
                n_states=n_states,
                n_inputs=n_inputs,
            )
        )

        self._plot_states_on_axes(state_axes_used, time, x, x_ref, x_labels)
        self._plot_inputs_on_axes(input_axes_used, time[:-1], u, u_labels)

        self._hide_inner_xlabels_block(
            axes_grid=state_axes_grid,
            n_cols=self.num_columns_states,
        )
        self._hide_inner_xlabels_block(
            axes_grid=input_axes_grid,
            n_cols=self.num_columns_inputs,
        )
        self._keep_only_bottom_xlabel(input_axes_grid)

        groups = get_shared_ylim_groups_state(self.drone_dim, self.only_positions)
        apply_shared_ylims(
            state_axes_grid,
            x_ref,
            x,
            groups_1based=groups,
            pad_frac=0.05,
        )

        save_figure(fig, self.plot_dir, "states_and_inputs.pdf")

    def _get_displayed_state_data(self):
        x = self.results.x_data.traj
        x_ref = self.results.x_data.ref_traj
        labels = drone.get_x_labels(self.drone_dim, self.only_positions)

        if self.only_positions:
            x_dim_displayed = x.shape[1] // 2
            x = x[:, :x_dim_displayed]
            x_ref = x_ref[:, :x_dim_displayed]
            labels = labels[:x_dim_displayed]
        else:
            x_dim_displayed = x.shape[1]

        return x, x_ref, labels, x_dim_displayed

    @staticmethod
    def _plot_states_on_axes(axes_used, time, x, x_ref, labels) -> None:
        first = True
        for i, ax in enumerate(axes_used):
            plot_x(
                [ax],
                time,
                [labels[i]],
                x[:, i:i + 1],
                x_ref[:, i:i + 1],
                show_legend=first,
            )
            first = False

    def _plot_inputs_on_axes(self, axes_used, time, u, labels) -> None:
        plot_u(axes_used, time, u, labels, self.drone_dim)

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