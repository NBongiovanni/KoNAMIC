from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol, TypeAlias

import numpy as np

from KoNAMIC import config
from KoNAMIC.core.control.config import KlqrControllerConfig, KmpcControllerConfig
from KoNAMIC.core.control.controllers import (
    BaseController,
    KoopmanLQRController,
    KoopmanMPCController,
    build_default_operating_input,
)
from KoNAMIC.core.control.mpc_solver import AcadosMPCSolver
from KoNAMIC.core.models import SensorKoopModel, VisionKoopModel
from KoNAMIC.core.models.model_config import ModelConfig
from KoNAMIC.core.plants import Plant
from KoNAMIC.core.scaling import DatasetScalers
from KoNAMIC.core.scenarios import ScenarioGenerator
from KoNAMIC.core.simulation import ClosedLoopTrajectory, KoopmanClosedLoopSimulator
from KoNAMIC.core.systems import SystemSpec
from KoNAMIC.utils.io import redirect_stdout_stderr_fd

from .pipeline import run_closed_loop_simulations


KoopmanClosedLoopControllerConfig: TypeAlias = KmpcControllerConfig | KlqrControllerConfig


class ClosedLoopRolloutConfig(Protocol):
    @property
    def num_rollouts(self) -> int: ...

    @property
    def dt(self) -> float: ...

    @property
    def t_sim(self) -> float: ...


def run_koopman_closed_loop_rollouts(
    *,
    modality: config.Modality,
    controller_dir: Path,
    solver_log_path: Path,
    model_config: ModelConfig,
    controller_config: KoopmanClosedLoopControllerConfig,
    closed_loop_config: ClosedLoopRolloutConfig,
    koop_model: SensorKoopModel | VisionKoopModel,
    system_spec: SystemSpec,
    scalers: DatasetScalers,
    scenario_generator: ScenarioGenerator,
    plant: Plant,
) -> list[ClosedLoopTrajectory]:
    with os.fdopen(os.dup(2), "w", closefd=True) as progress_file:
        with redirect_stdout_stderr_fd(solver_log_path):
            controller = _build_koopman_closed_loop_controller(
                modality=modality,
                controller_dir=controller_dir,
                model_config=model_config,
                controller_config=controller_config,
                koop_model=koop_model,
                scalers=scalers,
            )

            simulator = KoopmanClosedLoopSimulator(
                system_spec=system_spec,
                plant=plant,
                controller=controller,
                eval_config=closed_loop_config,
            )

            return run_closed_loop_simulations(
                num_simulations=closed_loop_config.num_rollouts,
                controller=controller,
                scenario_generator=scenario_generator,
                simulator=simulator,
                u_eq=build_default_operating_input(system_spec),
                progress_file=progress_file,
            )


def _build_koopman_closed_loop_controller(
    *,
    modality: config.Modality,
    controller_dir: Path,
    model_config: ModelConfig,
    controller_config: KoopmanClosedLoopControllerConfig,
    koop_model: SensorKoopModel | VisionKoopModel,
    scalers: DatasetScalers,
) -> BaseController:
    if isinstance(controller_config, KmpcControllerConfig):
        return KoopmanMPCController(
            modality=modality,
            model_config=model_config,
            controller_config=controller_config,
            solver=AcadosMPCSolver(controller_dir, controller_config),
            koop_model=koop_model,
            scalers=scalers,
            control_run_dir=controller_dir,
        )

    if isinstance(controller_config, KlqrControllerConfig):
        u_min, u_max = _build_klqr_input_bounds(controller_config, int(koop_model.u_dim))
        return KoopmanLQRController(
            model_config=model_config,
            koop_model=koop_model,
            scalers=scalers,
            q_diag=np.asarray(controller_config.q_diag, dtype=float),
            r_diag=np.asarray(controller_config.r_diag, dtype=float),
            u_min=u_min,
            u_max=u_max,
        )

    raise TypeError(
        "Unsupported Koopman closed-loop controller config: "
        f"{type(controller_config).__name__}. Expected KmpcControllerConfig or "
        "KlqrControllerConfig."
    )


def _build_klqr_input_bounds(
    controller_config: KlqrControllerConfig,
    u_dim: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    constraints = controller_config.constraints
    if not constraints.use_inputs_constraints:
        return None, None

    force_limits = constraints.force_limits
    torque_limits = constraints.torque_limits
    if torque_limits is None:
        raise ValueError("controller.constraints.torque_limits is required for KLQR.")

    if u_dim == 2:
        u_min = np.array([force_limits[0], torque_limits[0]], dtype=float)
        u_max = np.array([force_limits[1], torque_limits[1]], dtype=float)
    elif u_dim == 4:
        u_min = np.array(
            [force_limits[0], torque_limits[0], torque_limits[0], torque_limits[0]],
            dtype=float,
        )
        u_max = np.array(
            [force_limits[1], torque_limits[1], torque_limits[1], torque_limits[1]],
            dtype=float,
        )
    else:
        raise ValueError(f"Unsupported u_dim={u_dim} for KLQR input bounds.")

    return u_min, u_max

