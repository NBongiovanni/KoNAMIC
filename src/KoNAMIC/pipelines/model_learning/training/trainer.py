from __future__ import annotations
from pathlib import Path
from typing import TypeAlias
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from KoNAMIC.core.drone import DroneSpec
from KoNAMIC.core.models import SensorKoopModel as KoopModel
from KoNAMIC.core.models import SensorValForwardOutputs, TrainingContext

from ..losses.compute import SensorLossComputer, mean_sub_losses, SensorFullLoss, SensorsSubLosses
from ..losses.logger import TrainingLogger
from ..evaluation.evaluator import ModelEvaluator
from ..evaluation.closed_loop_replay import ClosedLoopReplayBuffer, ReplayConfig
from .config import PredictionHorizon
from .checkpoint_manager import CheckpointManager

BatchType: TypeAlias = Sequence[Tensor]


class Trainer:
    def __init__(
        self,
        modality: str,
        model: KoopModel,
        training_ctx: TrainingContext,
        data_loaders: dict,
        prediction_horizon: PredictionHorizon,
        drone: DroneSpec,
        x_scaler: StandardScaler,
        u_scaler: StandardScaler,
        model_params: dict,
        training_params: dict,
        control_params: dict,
        run_dir: Path,
    ) -> None:

        self.modality = modality
        self.model_params = model_params
        self.control_params = control_params
        self.training_params = training_params
        self.koop_model = model
        self.data_loaders = data_loaders
        self.drone = drone

        self.prediction_horizon = prediction_horizon
        self.num_steps_train = prediction_horizon.train
        self.num_val_datasets = len(prediction_horizon.val)

        self.optimizer = training_ctx.optimizer
        self.scheduler = training_ctx.scheduler
        self.writer = training_ctx.writer
        self.x_scaler = x_scaler
        self.u_scaler = u_scaler

        self.device = next(self.koop_model.parameters()).device
        self.grad_clip_max_norm = float(training_params.get("grad_clip_max_norm", 1.0))

        self.current_epoch = 0
        self.loss_computer = SensorLossComputer(training_params["loss_weights"])

        self.logger = TrainingLogger(
            modality=self.modality,
            writer=self.writer,
            num_val_datasets=self.num_val_datasets,
        )

        self.checkpoint_manager = CheckpointManager(
            checkpoints_dir=run_dir / "checkpoints",
            checkpoint_every=int(training_params["checkpoint_every"]),
        )

        self.evaluator = ModelEvaluator(
            modality=self.modality,
            model_params=self.model_params,
            control_params=self.control_params,
            drone=self.drone,
            koop_model=self.koop_model,
            data_loaders=self.data_loaders,
            device=self.device,
            prediction_horizon=self.prediction_horizon,
            reduce_sub_losses_fn=self.reduce_sub_losses,
            forward_and_loss_fn=self.forward_and_loss,
            closed_loop_every=training_params["closed_loop_every"],
            x_scaler=self.x_scaler,
            u_scaler=self.u_scaler,
        )

        self.closed_loop_replay = ClosedLoopReplayBuffer(
            x_scaler=self.x_scaler,
            u_scaler=self.u_scaler,
            num_steps=self.num_steps_train,
            config=ReplayConfig(
                enabled=training_params.get("closed_loop_replay_enabled", True),
                max_num_trajectories=training_params.get("closed_loop_replay_max_traj", 50),
                batch_size=training_params.get("closed_loop_replay_batch_size", 32),
            ),
        )

    def train_model(self) -> None:
        for epoch in range(self.training_params["num_epochs"]):
            self.current_epoch = epoch
            train_loss, train_sub_losses, mean_grad_norm = self.train_one_epoch()
            train_result = train_loss, train_sub_losses

            eval_result = self.evaluator.evaluate_epoch(epoch)

            self.closed_loop_replay.add(eval_result.closed_loop_trajectories)

            phase_losses = {"train": train_result, **eval_result.to_phase_losses_dict()}
            self.logger.log_losses(
                epoch=self.current_epoch,
                phase_losses=phase_losses,
                closed_loop_metrics=eval_result.closed_loop_metrics,
            )

            self.logger.log_training_state(
                epoch=self.current_epoch,
                optimizer=self.optimizer,
                total_norm=mean_grad_norm,
            )

            if self.checkpoint_manager.should_save(epoch):
                self.checkpoint_manager.save(
                    epoch=self.current_epoch,
                    model=self.koop_model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                )
        self.logger.close()

    def train_one_epoch(self) -> tuple[float, SensorsSubLosses, float]:
        self.koop_model.train()

        full_loss_arr: list[float] = []
        sub_losses_arr: list[SensorsSubLosses] = []
        total_norm_arr: list[float] = []

        self._train_on_loader(
            loader=self.data_loaders["train"],
            full_loss_arr=full_loss_arr,
            sub_losses_arr=sub_losses_arr,
            total_norm_arr=total_norm_arr,
        )

        replay_loader = self.closed_loop_replay.make_dataloader()
        if replay_loader is not None:
            self._train_on_loader(
                loader=replay_loader,
                full_loss_arr=full_loss_arr,
                sub_losses_arr=sub_losses_arr,
                total_norm_arr=total_norm_arr,
            )

        self.scheduler.step()

        full_mean = float(np.mean(full_loss_arr)) if full_loss_arr else float("nan")
        sub_mean = self.reduce_sub_losses(sub_losses_arr)
        mean_grad_norm = float(np.mean(total_norm_arr)) if total_norm_arr else float("nan")
        return full_mean, sub_mean, mean_grad_norm

    def forward_and_loss(self, batch: BatchType, num_steps: int) -> SensorFullLoss:
        x_gt_scaled, u_traj_scaled = (x.to(self.device, non_blocking=True) for x in batch)

        rec, pred = self.koop_model.forward(
            x_gt_scaled[:, 0],
            u_traj_scaled,
            num_steps,
        )
        z_proj = self.koop_model.batch_projection(x_gt_scaled)

        model_outputs = SensorValForwardOutputs(
            rec=rec,
            pred=pred,
            proj=z_proj,
            state_gt_scaled=x_gt_scaled,
            inputs_scaled=u_traj_scaled,
        )
        return self.loss_computer.compute(model_outputs, self.device)

    @staticmethod
    def reduce_sub_losses(sub_losses_arr: list[SensorsSubLosses]) -> SensorsSubLosses:
        return mean_sub_losses(sub_losses_arr)

    def _train_on_loader(
            self,
            loader,
            full_loss_arr: list[float],
            sub_losses_arr: list[SensorsSubLosses],
            total_norm_arr: list[float],
    ) -> None:
        for batch in tqdm(loader):
            self.optimizer.zero_grad(set_to_none=True)

            full_loss, sub_losses = self.forward_and_loss(
                batch=batch,
                num_steps=self.num_steps_train,
            )

            full_loss.backward()

            total_norm = torch.nn.utils.clip_grad_norm_(
                self.koop_model.parameters(),
                self.grad_clip_max_norm,
            )

            self.optimizer.step()

            full_loss_arr.append(float(full_loss.detach().cpu().item()))
            sub_losses_arr.append(sub_losses)
            total_norm_arr.append(float(total_norm.detach().cpu().item()))