from __future__ import annotations

import contextlib
import os

from sklearn.preprocessing import StandardScaler

from KoNAMIC.core import utils
from KoNAMIC.core.drone import DroneSpec
from KoNAMIC.core.plants import build_quad_plant
from KoNAMIC.core.models import SensorKoopModel
from KoNAMIC.core.simulation import ClosedLoopTrajectory
from KoNAMIC.core.control.controllers import KoopmanMPCController
from KoNAMIC.core.control.mpc_core import AcadosBackend
from KoNAMIC.pipelines.closed_loop_simulation import (
    create_reference_builder_factory,
    run_closed_loop_simulations,
    build_default_operating_input,
)
from KoNAMIC.pipelines.closed_loop_simulation.simulator.real import RealControlSimulator


class ClosedLoopEval:
    def __init__(
        self,
        modality: str,
        current_epoch: int,
        run_paths: utils.RunPaths,
        model_params: dict,
        control_params: dict,
        koop_model: SensorKoopModel,
        drone: DroneSpec,
        u_scaler: StandardScaler,
        x_scaler: StandardScaler,
    ) -> None:

        self.modality = modality
        self.current_epoch = current_epoch
        self.run_paths = run_paths
        self.model_params = model_params
        self.control_params = control_params
        self.koop_model = koop_model
        self.drone = drone
        self.u_scaler = u_scaler
        self.x_scaler = x_scaler
        self.plant = build_quad_plant(self.drone, self.model_params["dt"])

    def run_simulation(self) -> list[ClosedLoopTrajectory]:
        with suppress_stdout_stderr_fd():
            controller_dir = self.run_paths.training_eval_dir("closed_loop", self.current_epoch)
            controller = KoopmanMPCController(
                model_params=self.model_params,
                control_params=self.control_params,
                solver_backend=AcadosBackend(controller_dir, self.control_params),
                koop_model=self.koop_model,
                u_scaler=self.u_scaler,
                x_scaler=self.x_scaler,
            )
            simulator = RealControlSimulator(self.control_params, self.plant, controller)

            references_factory = create_reference_builder_factory(
                controller_type="koopman_mpc",
                modality=self.modality,
                control_params=self.control_params,
                drone=self.drone,
                koop_model=self.koop_model,
                x_scaler=self.x_scaler,
            )
            u_eq = build_default_operating_input(self.drone)

            simulation_results = run_closed_loop_simulations(
                num_simulations=self.control_params["num_simulations"],
                control_params=self.control_params,
                controller=controller,
                simulator=simulator,
                drone=self.drone,
                reference_builder_factory=references_factory,
                u_eq=u_eq,
            )
        return simulation_results


@contextlib.contextmanager
def suppress_stdout_stderr_fd():
    with open(os.devnull, "w") as devnull:
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)

        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)