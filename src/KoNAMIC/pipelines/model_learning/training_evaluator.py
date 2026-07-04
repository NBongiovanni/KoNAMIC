from __future__ import annotations

from collections.abc import Mapping

from torch.utils.data import DataLoader

from KoNAMIC import paths
from KoNAMIC.config import Modality
from KoNAMIC.core.scenarios import ScenarioGenerator
from KoNAMIC.core.simulation import compute_closed_loop_metrics
from KoNAMIC.core.models import SensorKoopModel, VisionKoopModel
from KoNAMIC.core.systems import SystemSpec
from KoNAMIC.core.plants import Plant
from KoNAMIC.pipelines.closed_loop_simulation import ClosedLoopEvaluator
from KoNAMIC.core.scaling import DatasetScalers
from KoNAMIC.pipelines.open_loop_simulation.evaluator import OpenLoopEvaluator
from KoNAMIC.pipelines.data_preparation.data_loaders import (
    DataLoaderSplit,
    PreparedDataLoaders,
)

from .losses.classes import EpochEvalResult
from .losses.compute_vision import VisionLossComputer
from .losses.compute_sensor import SensorLossComputer
from .config import TrainingPipelineConfig
from .training.curriculum import CurriculumManager
from .training.forward_loss_computer import build_forward_loss_computer


class TrainingEvaluator:
    """
    Orchestre les évaluations réalisées pendant l'entraînement.

    Responsabilités :
    - lancer les splits de validation en open loop ;
    - décider si l'évaluation closed loop doit être lancée ;
    - calculer les métriques closed loop ;
    - retourner un EpochEvalResult compatible avec le Trainer actuel.
    """

    def __init__(
        self,
        *,
        modality: Modality,
        system_spec: SystemSpec,
        run_config: TrainingPipelineConfig,
        run_paths: paths.RunPaths,
        koop_model: SensorKoopModel | VisionKoopModel,
        data_loaders: PreparedDataLoaders | Mapping[str, DataLoader],
        scalers: DatasetScalers,
        scenario_generator: ScenarioGenerator,
        plant: Plant,
    ) -> None:
        """Initialize the evaluator used during model training.

        This object centralizes validation-time evaluation while keeping it
        separate from optimizer updates in Trainer. It builds and owns the
        open-loop evaluator for rollout losses on validation splits, and the
        closed-loop evaluator for controller-in-the-loop checks scheduled by
        the training config. The evaluator receives already-built systems,
        experiments, scalers, data loaders, scenario generators, and plants, so it
        does not create training data or mutate the learning pipeline structure.
        Its public output is an EpochEvalResult consumed by Trainer logging and
        checkpoint decisions.
        """

        self.modality = modality
        self.current_epoch = 0

        self.run_paths = run_paths
        self.run_config = run_config
        self.system_spec = system_spec
        self.koop_model = koop_model
        self.data_loaders = data_loaders
        self.scalers = scalers
        self.scenario_generator = scenario_generator
        self.plant = plant

        self.model_config = run_config.model
        self.trainer_config = run_config.trainer
        self.controller_config = run_config.closed_loop_eval_controller
        self.open_loop_eval_config = run_config.open_loop_eval
        self.closed_loop_eval_config = run_config.closed_loop_eval

        self.validation_items = self._build_validation_items()
        self._validate_validation_horizons()
        self.curriculum: CurriculumManager | None = None
        if self.modality is Modality.VISION and self.trainer_config.curriculum is not None:
            self.curriculum = CurriculumManager(self.trainer_config.curriculum)

        loss_computer = self._build_loss_computer()

        forward_loss_computer = build_forward_loss_computer(
            modality=self.modality,
            koop_model=self.koop_model,
            loss_computer=loss_computer,
            phases_active=self._vision_phases_active,
            effective_weight=self._vision_weight_fn,
        )

        self.open_loop_evaluator = OpenLoopEvaluator(
            forward_loss_computer=forward_loss_computer,
            data_loaders=self.data_loaders,
            scalers=self.scalers,
            system_spec=self.system_spec,
            run_paths=self.run_paths,
            model_config=self.model_config,
            open_loop_eval_config=self.open_loop_eval_config,
            modality=self.modality,
        )

        self.closed_loop_evaluator = ClosedLoopEvaluator(
            modality=self.modality,
            run_paths=self.run_paths,
            model_config=self.model_config,
            controller_config=self.controller_config,
            closed_loop_eval_config=self.closed_loop_eval_config,
            koop_model=self.koop_model,
            system_spec=self.system_spec,
            scalers=self.scalers,
            scenario_generator=self.scenario_generator,
            plant=self.plant,
        )

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch
        if self.curriculum is not None:
            self.curriculum.maybe_activate_phases(epoch)

    def _vision_weight_fn(self, base: float, key: str) -> float:
        if self.curriculum is None:
            return base
        return self.curriculum.effective_weight(base, key, self.current_epoch)

    def _vision_phases_active(self) -> list[bool]:
        if self.curriculum is None:
            return [False]
        return self.curriculum.phases_active

    def evaluate_epoch(self, epoch: int) -> EpochEvalResult:
        self.set_epoch(epoch)

        open_loop_losses = {}
        for item in self.validation_items:
            open_loop_losses[item.split] = self.open_loop_evaluator.evaluate(
                phase=item.split,
                num_steps=self._validation_horizon(item),
            )

        if self.should_render_open_loop_rollouts(epoch):
            render_item = self.validation_items[-1]
            self.open_loop_evaluator.render_sample_rollouts(
                epoch=epoch,
                phase=render_item.split,
                num_steps=self.open_loop_eval_config.num_steps_simulation,
                only_position=False,
            )

        if self.should_run_closed_loop(epoch):
            closed_loop_result = self.closed_loop_evaluator.evaluate(epoch=epoch)

            state_names = self.system_spec.get_x_names(only_positions=False)
            closed_loop_metrics = compute_closed_loop_metrics(
                results=closed_loop_result,
                state_labels=state_names,
            )
        else:
            closed_loop_result = None
            closed_loop_metrics = None

        return EpochEvalResult(
            open_loop_losses=open_loop_losses,
            closed_loop_trajectories=closed_loop_result,
            closed_loop_metrics=closed_loop_metrics,
        )


    def _build_validation_items(self) -> tuple[DataLoaderSplit, ...]:
        if isinstance(self.data_loaders, PreparedDataLoaders):
            return self.data_loaders.validation_items

        return tuple(
            DataLoaderSplit(
                split=f"val_{idx + 1}",
                name=f"val_{idx + 1}",
                loader=self.data_loaders[f"val_{idx + 1}"],
                num_steps_pred=num_steps,
            )
            for idx, num_steps in enumerate(self.run_config.prediction_horizon.val)
        )

    def _validate_validation_horizons(self) -> None:
        configured_horizons = self.run_config.prediction_horizon.val
        if len(self.validation_items) != len(configured_horizons):
            raise ValueError(
                "Number of validation loaders does not match prediction horizons: "
                f"got {len(self.validation_items)} loaders and "
                f"{len(configured_horizons)} horizons."
            )

        for item, configured_horizon in zip(self.validation_items, configured_horizons):
            if item.num_steps_pred is None:
                continue
            if item.num_steps_pred != configured_horizon:
                raise ValueError(
                    f"Validation horizon mismatch for {item.split!r}: "
                    f"loader metadata has {item.num_steps_pred}, "
                    f"run config has {configured_horizon}."
                )

    def _validation_horizon(self, item: DataLoaderSplit) -> int:
        if item.num_steps_pred is not None:
            return item.num_steps_pred

        idx = self.validation_items.index(item)
        return self.run_config.prediction_horizon.val[idx]

    def should_render_open_loop_rollouts(self, epoch: int) -> bool:
        if not self.open_loop_eval_config.save_plots:
            return False

        plot_every = self.open_loop_eval_config.plot_every
        if plot_every <= 0:
            return False

        return epoch % plot_every == 0

    def should_run_closed_loop(self, epoch: int) -> bool:
        closed_loop_eval_every = self.trainer_config.closed_loop_eval_every
        start_epoch = (
            self.closed_loop_eval_config.start_epoch
            if self.closed_loop_eval_config.start_epoch is not None
            else 1
        )

        if closed_loop_eval_every is None or closed_loop_eval_every <= 0:
            return False

        if not self.closed_loop_eval_config.enabled:
            return False

        if epoch < start_epoch:
            return False

        return epoch % closed_loop_eval_every == 0

    def _build_loss_computer(self):
        if self.modality is Modality.SENSOR:
            state_rmse_units_scale = None
            if self.run_config.data_preparation.scaler.scale_x:
                state_rmse_units_scale = [
                    float(value) for value in self.scalers.x.scale_
                ]
            return SensorLossComputer(
                self.trainer_config.loss_weights,
                self.system_spec.get_x_names(),
                state_rmse_units_scale=state_rmse_units_scale,
            )

        if self.modality is Modality.VISION:
            return VisionLossComputer(
                self.trainer_config.loss_weights,
            )

        raise ValueError(f"Unknown modality: {self.modality}")
