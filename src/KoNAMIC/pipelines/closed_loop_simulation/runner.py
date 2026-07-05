from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol, TypeAlias

from KoNAMIC import config
from KoNAMIC.core.control.config import KlqrControllerConfig, KmpcControllerConfig
from KoNAMIC.core.control.controllers import (
    BaseController,
    build_default_operating_input,
)
from KoNAMIC.koopman.controllers import build_koopman_controller_for_dir
from KoNAMIC.koopman.models import SensorKoopModel, VisionKoopModel
from KoNAMIC.koopman.models.model_config import ModelConfig
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
    return build_koopman_controller_for_dir(
        modality=modality,
        controller_dir=controller_dir,
        model_config=model_config,
        controller_config=controller_config,
        koop_model=koop_model,
        data_scalers=scalers,
    )
