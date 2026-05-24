from pathlib import Path
import pickle
from types import SimpleNamespace
from typing import Sequence

import numpy as np
from scipy.io import loadmat

from KoNAMIC.core import drone
from KoNAMIC.viz import (
    plot_u_multi,
    plot_state_multi,
    save_figure,
    apply_shared_ylims,
    get_shared_ylim_groups_state,
)

from .base_visualizer import BaseClosedLoopVisualizer


class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        module = module.replace("KSIC_v6", "KoNAMIC")
        module = module.replace("KSIC_v8", "KoNAMIC")
        module = module.replace("closed_loop_eval", "pipelines.closed_loop_simulation")
        module = module.replace("pipelines.closed_loop_simulation.simulation", "pipelines.closed_loop_simulation.simulator")
        module = module.replace("pipelines.closed_loop_simulation.simulator.results", "pipelines.closed_loop_simulation.results")
        return super().find_class(module, name)


class ClosedLoopMultiVisualizer(BaseClosedLoopVisualizer):
    def __init__(
        self,
        drone_dim: int,
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
            drone_dim=drone_dim,
            dt=dt,
            only_position=only_position,
            num_columns_states=num_columns_states,
            num_columns_inputs=num_columns_inputs,
        )

        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        self.names = list(names)
        self.colors = list(colors)
        self.filename = filename
        self.results_list = None

        self.x_dim, self.u_dim, self.x_ref_dim = drone.get_dimensions(drone_dim)

    def visualize(self) -> None:
        assert self.results_list is not None
        self.run_with_rc_context()

    def load_results(self, *paths: Path) -> None:
        loaded = []

        for p in paths:
            p = Path(p)

            if p.is_file() and p.suffix == ".pkl":
                loaded.append(self.load_sim_result(p))

            elif p.is_dir():
                required = ["sensor.mat", "refs.mat", "inputs.mat", "time.mat"]
                if all((p / name).exists() for name in required):
                    loaded.append(self.load_pid_result(p))
                else:
                    raise ValueError(
                        f"Directory {p} is not a valid PID result folder "
                        f"(missing one of {required})"
                    )
            else:
                raise ValueError(f"Unsupported result path: {p}")

        self.results_list = loaded

    def _plot(self) -> None:
        labels_x = drone.get_x_labels(self.drone_dim, self.only_position)
        labels_u = drone.get_u_labels(self.drone_dim)

        runs_x = []
        for name, r in zip(self.names, self.results_list):
            runs_x.append((r.time, r.x_data.ref_traj, r.x_data.traj, name))

        runs_u = []
        for name, r in zip(self.names, self.results_list):
            t_u = r.time[:-1]
            runs_u.append((t_u, r.inputs_data.u_physical, name))

        x_dim_disp = len(labels_x)
        _, x_ref0, x_traj0, _ = runs_x[0]

        if x_ref0.shape[1] != x_dim_disp:
            runs_x = [
                (t, x_ref[:, :x_dim_disp], x_traj[:, :x_dim_disp], name)
                for (t, x_ref, x_traj, name) in runs_x
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
            state_axes_grid=state_axes_grid,
            state_axes_used=state_axes_used,
            runs_x=runs_x,
            x_labels=labels_x,
            n_ref_disp=n_ref_disp,
        )

        self._plot_inputs_on_axes(
            input_axes_used=input_axes_used,
            runs_u=runs_u,
            u_labels=labels_u,
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

    def _plot_states_on_axes(
        self,
        state_axes_grid,
        state_axes_used,
        runs_x,
        x_labels,
        n_ref_disp: int,
    ) -> None:
        plot_state_multi(
            state_axes_used,
            len(x_labels),
            runs_x,
            self.names,
            x_labels,
            self.colors,
            n_ref_disp,
            "Reference",
            "MAE",
            show_metric=True,
        )

        groups = get_shared_ylim_groups_state(self.drone_dim, self.only_position)
        x_ref_all = np.concatenate([r[1] for r in runs_x], axis=0)
        x_traj_all = np.concatenate([r[2] for r in runs_x], axis=0)

        apply_shared_ylims(
            axes=state_axes_grid,
            x_gt=x_ref_all,
            x_pred=x_traj_all,
            groups_1based=groups,
            pad_frac=0.05,
        )

    def _plot_inputs_on_axes(
        self,
        input_axes_used,
        runs_u,
        u_labels,
    ) -> None:
        plot_u_multi(
            drone_dim=self.drone_dim,
            axes=input_axes_used,
            runs_u=runs_u,
            u_labels=u_labels,
            colors=self.colors,
            grouped_ylabel=r"$\tau$ [N.m]",
            group_legend_labels=[r"$\tau_1$", r"$\tau_2$", r"$\tau_3$"],
        )

    @staticmethod
    def load_sim_result(path: Path):
        with open(path, "rb") as f:
            return RenameUnpickler(f).load()

    def load_pid_result(self, pid_dir: Path):
        states = loadmat(pid_dir / "sensor.mat")["sensor"]
        refs = loadmat(pid_dir / "refs.mat")["statesRef"]
        inputs = loadmat(pid_dir / "inputs.mat")["inputs"]
        time = loadmat(pid_dir / "time.mat")["timeVec"]

        angles_indexes = drone.get_angle_indexes(self.drone_dim)

        time = np.squeeze(time)
        refs = np.squeeze(refs)
        inputs = np.squeeze(inputs)
        states = np.squeeze(states)

        states[:, angles_indexes] = np.rad2deg(states[:, angles_indexes])

        if len(states) != len(refs):
            raise ValueError(
                f"sensor and refs must have same length, got {states.shape} and {refs.shape}"
            )

        if len(time) != len(states):
            raise ValueError(
                f"time and sensor must have same length, got {time.shape} and {states.shape}"
            )

        inputs = inputs[:-1]

        refs_full = np.zeros((refs.shape[0], 12))
        refs_full[:, :6] = refs

        return SimpleNamespace(
            time=time,
            x_data=SimpleNamespace(
                ref_traj=refs_full,
                traj=states,
            ),
            inputs_data=SimpleNamespace(
                u_physical=inputs,
            ),
        )