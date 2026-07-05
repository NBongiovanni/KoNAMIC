from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from KoNAMIC import paths
from KoNAMIC.config import Modality
from KoNAMIC.core.scaling import DatasetScalers
from KoNAMIC.koopman.models import (
    SensorKoopModel,
    VisionKoopModel,
)
from KoNAMIC.pipelines.data_preparation.data_loaders import PreparedDataLoaders
from .context import TrainingContext
from KoNAMIC.core.systems import SystemSpec
from KoNAMIC.pipelines.model_learning.closed_loop_replay import ClosedLoopReplayBuffer, ReplayConfig
from KoNAMIC.koopman.training.losses import build_loss_computer, reduce_sub_losses, SubLosses
from KoNAMIC.koopman.training.losses.classes import OpenLoopLosses
from KoNAMIC.pipelines.model_learning.reporting import TrainingLogger, build_metrics_backend
from .checkpoint_manager import CheckpointManager
from .curriculum import CurriculumManager
from KoNAMIC.pipelines.model_learning.config import TrainingPipelineConfig
from .forward_loss_computer import build_forward_loss_computer
from KoNAMIC.pipelines.model_learning.closed_loop_augmenter import ClosedLoopAugmenter
from KoNAMIC.pipelines.model_learning.closed_loop_trajectory_generator import ClosedLoopTrajectoryGenerator

if TYPE_CHECKING:
    from KoNAMIC.pipelines.model_learning.training_evaluator import TrainingEvaluator

BatchType: TypeAlias = Sequence[Tensor]
KoopModel: TypeAlias = SensorKoopModel | VisionKoopModel


class Trainer:
    def __init__(
        self,
        modality: Modality,
        system_spec: SystemSpec,
        run_config: TrainingPipelineConfig,
        run_paths: paths.RunPaths,
        koop_model: KoopModel,
        training_ctx: TrainingContext,
        data_loaders: PreparedDataLoaders | Mapping[str, DataLoader],
        scalers: DatasetScalers,
        model_evaluator: TrainingEvaluator,
    ) -> None:
        """Initialize the training orchestrator for one Koopman learning run.

        The trainer owns epoch scheduling, batch iteration, loss computation,
        optimizer steps, checkpointing, and metric logging. It delegates
        rollout-quality evaluation to TrainingEvaluator instead of mixing
        evaluation policy into the optimization loop. Optional closed-loop data
        augmentation is delegated to a dedicated augmenter so generated
        trajectories remain separate from the base datasets. Sensor and vision
        training share this high-level loop through typed configs and helper
        objects. Model dimensions, scalers, loaders, and controllers are
        expected to be resolved before construction. Keep reusable math, losses,
        simulation, and controller logic in their dedicated modules.
        """

        self.modality = modality
        self.run_paths = run_paths
        self.trainer_config = run_config.trainer
        self.closed_loop_training_config = run_config.closed_loop_training

        self.koop_model = koop_model
        self.data_loaders = data_loaders
        self.system_spec = system_spec
        self.evaluator = model_evaluator

        self.num_steps_train = run_config.prediction_horizon.train
        self.num_val_datasets = len(run_config.prediction_horizon.val)

        self.optimizer = training_ctx.optimizer
        self.scheduler = training_ctx.scheduler
        self.writer = training_ctx.writer
        self.scalers = scalers
        self.grad_clip_max_norm = self.trainer_config.grad_clip_max_norm
        self.current_epoch = 0

        self.loss_computer = build_loss_computer(
            modality=self.modality.key,
            loss_weights=self.trainer_config.loss_weights,
            system_spec=self.system_spec,
            x_scaler=self.scalers.x,
            scale_x=run_config.data_preparation.scaler.scale_x,
        )

        self.curriculum: CurriculumManager | None = None
        if self.modality is Modality.VISION and self.trainer_config.curriculum is not None:
            self.curriculum = CurriculumManager(self.trainer_config.curriculum)

        logging_backend = build_metrics_backend(
            writer=self.writer,
            logging_config=self.trainer_config.logging,
            run_name=self.run_paths.run_dir.name,
            run_config=run_config.to_dict(),
        )
        logging_backend.log_config(run_config.to_dict())

        self.logger = TrainingLogger(
            modality=self.modality.key,
            backend=logging_backend,
            num_val_datasets=self.num_val_datasets,
            system_spec=self.system_spec,
        )

        self.checkpoint_manager = CheckpointManager(
            checkpoints_dir=self.run_paths.checkpoints_dir,
            checkpoint_every=int(self.trainer_config.checkpoint_every),
        )
        self.closed_loop_augmenter = self._build_closed_loop_augmenter()
        self.forward_loss_computer = build_forward_loss_computer(
            modality=self.modality,
            koop_model=self.koop_model,
            loss_computer=self.loss_computer,
            phases_active=self._vision_phases_active,
            effective_weight=self._vision_weight_fn,
        )

    def train_model(self) -> None:
        for epoch in range(self.trainer_config.num_epochs):
            self.current_epoch = epoch
            self.evaluator.set_epoch(epoch)
            train_loss, train_sub_losses, mean_grad_norm = self.train_one_epoch()
            train_result = OpenLoopLosses(
                full_loss=train_loss,
                sub_losses=train_sub_losses,
            )

            eval_result = self.evaluator.evaluate_epoch(epoch)
            if self.closed_loop_augmenter is not None:
                self.closed_loop_augmenter.maybe_generate(epoch=epoch)

            phase_losses = {"train": train_result, **eval_result.to_phase_losses_dict()}
            self.logger.log_losses(
                epoch=self.current_epoch,
                phase_losses=phase_losses,
                closed_loop_metrics=eval_result.closed_loop_metrics,
            )

            self._log_training_state(mean_grad_norm)

            if self.checkpoint_manager.should_save(epoch):
                self.checkpoint_manager.save(
                    epoch=self.current_epoch,
                    model=self.koop_model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                )

        self.logger.close()

    def train_one_epoch(self) -> tuple[float, SubLosses, float]:
        self.koop_model.train()

        full_loss_arr: list[float] = []
        sub_losses_arr: list[SubLosses] = []
        total_norm_arr: list[float] = []

        self._train_on_loader(
            loader=self._train_loader(),
            full_loss_arr=full_loss_arr,
            sub_losses_arr=sub_losses_arr,
            total_norm_arr=total_norm_arr,
        )

        self._train_on_closed_loop_replay(
            full_loss_arr=full_loss_arr,
            sub_losses_arr=sub_losses_arr,
            total_norm_arr=total_norm_arr,
        )

        self.scheduler.step()

        full_mean = float(np.mean(full_loss_arr)) if full_loss_arr else float("nan")
        sub_mean = reduce_sub_losses(sub_losses_arr)
        mean_grad_norm = float(np.mean(total_norm_arr)) if total_norm_arr else float("nan")
        return full_mean, sub_mean, mean_grad_norm

    def _train_loader(self) -> DataLoader:
        if isinstance(self.data_loaders, PreparedDataLoaders):
            return self.data_loaders.train.loader

        return self.data_loaders["train"]

    def _train_on_closed_loop_replay(
            self,
            full_loss_arr: list[float],
            sub_losses_arr: list[SubLosses],
            total_norm_arr: list[float],
    ) -> None:
        if self.closed_loop_augmenter is None:
            return

        replay_loader = self.closed_loop_augmenter.make_dataloader(
            epoch=self.current_epoch,
        )
        if replay_loader is None:
            return

        self._train_on_loader(
            loader=replay_loader,
            full_loss_arr=full_loss_arr,
            sub_losses_arr=sub_losses_arr,
            total_norm_arr=total_norm_arr,
        )

    def _vision_weight_fn(self, base: float, key: str) -> float:
        if self.curriculum is None:
            return base
        return self.curriculum.effective_weight(base, key, self.current_epoch)

    def _vision_phases_active(self) -> list[bool]:
        if self.curriculum is None:
            return [False]
        return self.curriculum.phases_active

    def _build_closed_loop_augmenter(self) -> ClosedLoopAugmenter | None:
        replay_config = ReplayConfig(self.closed_loop_training_config)

        if not replay_config.enabled:
            return None

        if self.modality is not Modality.SENSOR:
            print(
                "[WARNING] closed_loop_training.enabled=True ignored for vision. "
                "Closed-loop augmentation is currently sensor-only."
            )
            return None

        replay_buffer = ClosedLoopReplayBuffer(
            scalers=self.scalers,
            num_steps=self.num_steps_train,
            config=replay_config,
        )

        trajectory_generator = ClosedLoopTrajectoryGenerator(
            modality=self.modality,
            run_paths=self.run_paths,
            model_config=self.evaluator.model_config,
            controller_config=self.evaluator.run_config.closed_loop_training_controller,
            closed_loop_training_config=self.closed_loop_training_config,
            koop_model=self.koop_model,
            system_spec=self.system_spec,
            scalers=self.scalers,
            scenario_generator=self.evaluator.scenario_generator,
            plant=self.evaluator.plant,
        )

        return ClosedLoopAugmenter(
            config=self.closed_loop_training_config,
            replay_buffer=replay_buffer,
            generator=trajectory_generator,
        )

    def _log_training_state(self, mean_grad_norm: float) -> None:
        if self.modality is Modality.SENSOR:
            self.logger.log_training_state(
                epoch=self.current_epoch,
                optimizer=self.optimizer,
                total_norm=mean_grad_norm,
            )
        elif self.modality is Modality.VISION:
            if self.curriculum is None:
                self.logger.log_training_state(
                    epoch=self.current_epoch,
                    optimizer=self.optimizer,
                    total_norm=mean_grad_norm,
                )
                return

            self.curriculum.maybe_activate_phases(self.current_epoch)

            self.logger.log_training_state(
                epoch=self.current_epoch,
                optimizer=self.optimizer,
                total_norm=mean_grad_norm,
                base=self.trainer_config.loss_weights,
                effective_weight=self._vision_weight_fn,
                phase_index=self.curriculum.current_phase_index() + 1,
            )
        else:
            raise ValueError(f"Unknown modality: {self.modality}")

    def _train_on_loader(
        self,
        loader,
        full_loss_arr: list[float],
        sub_losses_arr: list[SubLosses],
        total_norm_arr: list[float],
    ) -> None:
        for batch in tqdm(loader):
            self.optimizer.zero_grad(set_to_none=True)
            full_loss, sub_losses = self.forward_loss_computer.compute(
                batch=batch,
                num_steps=self.num_steps_train,
            )
            full_loss.backward()

            total_norm = torch.nn.utils.clip_grad_norm_(
                self.koop_model.parameters(),
                self.grad_clip_max_norm
                if self.grad_clip_max_norm is not None
                else float("inf"),
            )
            self.optimizer.step()

            full_loss_arr.append(float(full_loss.detach().cpu().item()))
            sub_losses_arr.append(sub_losses)
            total_norm_arr.append(float(total_norm.detach().cpu().item()))
