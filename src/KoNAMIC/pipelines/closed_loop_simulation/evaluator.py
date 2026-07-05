from __future__ import annotations

from KoNAMIC import paths, config
from KoNAMIC.core.control.config import KlqrControllerConfig, KmpcControllerConfig
from KoNAMIC.koopman.models.model_config import ModelConfig
from KoNAMIC.koopman.models.sensor_koop_model import SensorKoopModel
from KoNAMIC.koopman.models.vision_koop_model import VisionKoopModel
from KoNAMIC.core.plants import Plant
from KoNAMIC.core.scenarios import ScenarioGenerator
from KoNAMIC.core.simulation import ClosedLoopTrajectory, build_time_grid
from KoNAMIC.core.systems import SystemSpec
from KoNAMIC.core.scaling import DatasetScalers

from .config import ClosedLoopEvalConfig
from .runner import run_koopman_closed_loop_rollouts
from .viz.viz_pipeline import run_closed_loop_visualization


class ClosedLoopEvaluator:
    """
    Lance une évaluation closed loop pour un modèle Koopman donné.

    Cette classe ne décide pas quand lancer l'évaluation.
    Elle sait seulement l'exécuter pour une epoch donnée.
    """

    def __init__(
        self,
        *,
        modality: config.Modality,
        run_paths: paths.RunPaths,
        model_config: ModelConfig,
        controller_config: KmpcControllerConfig | KlqrControllerConfig,
        closed_loop_eval_config: ClosedLoopEvalConfig,
        koop_model: SensorKoopModel | VisionKoopModel,
        system_spec: SystemSpec,
        scalers: DatasetScalers,
        scenario_generator: ScenarioGenerator,
        plant: Plant,
    ) -> None:
        self.modality = modality
        self.run_paths = run_paths
        self.model_config = model_config
        self.controller_config = controller_config
        self.closed_loop_eval_config = closed_loop_eval_config
        self.koop_model = koop_model
        self.system_spec = system_spec
        self.scalers = scalers
        self.scenario_generator = scenario_generator
        self.plant = plant
        self.dt = closed_loop_eval_config.dt
        self.t_sim = closed_loop_eval_config.t_sim
        self.time = build_time_grid(self.dt, self.t_sim)

    def evaluate(self, *, epoch: int) -> list[ClosedLoopTrajectory]:
        controller_dir = self.run_paths.training_eval_dir("closed_loop", epoch)
        controller_dir.mkdir(parents=True, exist_ok=True)

        solver_log_path = controller_dir / "solver_stdout_stderr.log"
        print(f"[closed_loop] epoch={epoch} solver log: {solver_log_path}")

        simulation_results = run_koopman_closed_loop_rollouts(
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

        num_visualized = min(
            self.closed_loop_eval_config.num_visualized_rollouts,
            len(simulation_results),
        )
        run_closed_loop_visualization(
            simulation_indexes=list(range(num_visualized)),
            system_spec=self.system_spec,
            base_control_runs_dir=controller_dir,
            simulation_results=simulation_results,
            num_columns_states=2,
            num_columns_inputs=2,
            only_positions=False,
        )
        return simulation_results
