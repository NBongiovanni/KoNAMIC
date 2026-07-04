from __future__ import annotations

from KoNAMIC import paths, config
from KoNAMIC.core.control.config import KlqrControllerConfig, KmpcControllerConfig
from KoNAMIC.core.models.model_config import ModelConfig
from KoNAMIC.core.models.sensor_koop_model import SensorKoopModel
from KoNAMIC.core.plants import Plant
from KoNAMIC.core.scenarios import ScenarioGenerator
from KoNAMIC.core.simulation import ClosedLoopTrajectory, build_time_grid
from KoNAMIC.core.systems import SystemSpec
from KoNAMIC.core.scaling import DatasetScalers
from KoNAMIC.pipelines.closed_loop_simulation.runner import (
    run_koopman_closed_loop_rollouts,
)
from .config import ClosedLoopTrainingConfig


class ClosedLoopTrajectoryGenerator:
    """
    Execute des rollouts closed-loop pour un modele Koopman donne.

    Cette classe est utilisee pendant l entrainement lorsque la pipeline veut
    generer des trajectoires fermees par un controleur Koopman. Elle assemble
    les dependances deja construites, choisit le dossier de sortie associe a
    l epoch courante, cree le journal du solveur, puis delegue l execution
    effective des rollouts a la couche closed_loop_simulation.

    Elle ne decide pas quand lancer cette generation, combien elle doit peser
    dans l apprentissage, ni comment les trajectoires produites sont reinjectees
    dans les donnees d entrainement. Ces decisions restent dans la configuration
    de training et dans l augmenter. Ce decoupage garde la generation de
    trajectoires testable separement de la logique d optimisation.
    """

    def __init__(
        self,
        *,
        modality: config.Modality,
        run_paths: paths.RunPaths,
        model_config: ModelConfig,
        controller_config: KmpcControllerConfig | KlqrControllerConfig,
        closed_loop_training_config: ClosedLoopTrainingConfig,
        koop_model: SensorKoopModel,
        system_spec: SystemSpec,
        scalers: DatasetScalers,
        scenario_generator: ScenarioGenerator,
        plant: Plant,
    ) -> None:
        self.modality = modality
        self.run_paths = run_paths
        self.model_config = model_config
        self.controller_config = controller_config
        self.closed_loop_eval_config = closed_loop_training_config
        self.koop_model = koop_model
        self.system_spec = system_spec
        self.scalers = scalers
        self.scenario_generator = scenario_generator
        self.plant = plant
        self.dt = closed_loop_training_config.dt
        self.t_sim = closed_loop_training_config.t_sim
        self.time = build_time_grid(self.dt, self.t_sim)

    def generate(self, *, epoch: int) -> list[ClosedLoopTrajectory]:
        controller_dir = self.run_paths.training_eval_dir("closed_loop", epoch)
        controller_dir.mkdir(parents=True, exist_ok=True)

        solver_log_path = controller_dir / "augmentation_solver_stdout_stderr.log"
        print(f"[closed_loop_augmentation] epoch={epoch} solver log: {solver_log_path}")

        return run_koopman_closed_loop_rollouts(
            modality=self.modality,
            controller_dir=controller_dir,
            solver_log_path=solver_log_path,
            model_config=self.model_config,
            controller_config=self.controller_config,
            closed_loop_config=self.closed_loop_eval_config,
            koop_model=self.koop_model,
            system_spec=self.system_spec,
            scalers=self.scalers,
            scenario_generator=self.scenario_generator,
            plant=self.plant,
        )
