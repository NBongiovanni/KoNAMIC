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
from KoNAMIC.core.models import (
    SensorKoopModel,
    VisionKoopModel,
    SensorValForwardOutputs,
    VisionValForwardOutputs,
    TrainingContext,
)

from ..losses.compute import (
    SensorLossComputer,
    VisionLossComputer,
    mean_sub_losses,
)
from ..losses.logger import TrainingLogger
from ..evaluation.evaluator import ModelEvaluator
from ..evaluation.closed_loop_replay import ClosedLoopReplayBuffer, ReplayConfig
from ..evaluation.ground_truth import build_ground_truth_from_images
from .config import PredictionHorizon
from .checkpoint_manager import CheckpointManager
from .curriculum import CurriculumManager

BatchType: TypeAlias = Sequence[Tensor]
KoopModel: TypeAlias = SensorKoopModel | VisionKoopModel
SubLosses: TypeAlias = dict[str, float]


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

        # --- modality-specific objects ---
        self.curriculum = None
        if self.modality == "sensor":
            self.loss_computer = SensorLossComputer(training_params["loss_weights"])
        elif self.modality == "vision":
            self.curriculum = CurriculumManager(training_params["curriculum"])
            self.loss_computer = VisionLossComputer(training_params["loss_weights"])
        else:
            raise ValueError(f"Unknown modality: {self.modality}")

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

        # Pour ce commit minimal, le replay closed-loop reste sensor-only.
        # Sinon il risque de réinjecter des batchs (x, u) dans un modèle vision.
        if self.modality == "sensor":
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
        else:
            if training_params.get("closed_loop_replay_enabled", False):
                print(
                    "[WARNING] closed_loop_replay_enabled=True ignored for vision. "
                    "Closed-loop replay is currently sensor-only."
                )
            self.closed_loop_replay = None

    def train_model(self) -> None:
        for epoch in range(self.training_params["num_epochs"]):
            self.current_epoch = epoch
            train_loss, train_sub_losses, mean_grad_norm = self.train_one_epoch()
            train_result = train_loss, train_sub_losses

            eval_result = self.evaluator.evaluate_epoch(epoch)

            if self.closed_loop_replay is not None:
                self.closed_loop_replay.add(eval_result.closed_loop_trajectories)

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
            loader=self.data_loaders["train"],
            full_loss_arr=full_loss_arr,
            sub_losses_arr=sub_losses_arr,
            total_norm_arr=total_norm_arr,
        )

        if self.closed_loop_replay is not None:
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

    def forward_and_loss(self, batch: BatchType, num_steps: int):
        if self.modality == "sensor":
            return self.forward_and_loss_sensor(batch, num_steps)
        if self.modality == "vision":
            return self.forward_and_loss_vision(batch, num_steps)
        raise ValueError(f"Unknown modality: {self.modality}")

    def forward_and_loss_sensor(self, batch: BatchType, num_steps: int):
        x_gt_scaled, u_traj_scaled = (
            x.to(self.device, non_blocking=True) for x in batch
        )

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

    def forward_and_loss_vision(self, batch: BatchType, num_steps: int):
        """
        Expected vision batch convention from the old VisionTrainer:
            y_data: (B, T, C, H, W)
            u_data: (B, T, n_u)
            x_data: optional physical/geometric state, used for targets/diagnostics
        """
        if len(batch) != 3:
            shapes = [tuple(x.shape) for x in batch]
            raise ValueError(
                "Vision training expects batch=(y_data, u_data, x_data), "
                f"but got len(batch)={len(batch)}, shapes={shapes}"
            )

        y_data, u_data, x_data = (
            x.to(self.device, non_blocking=True) for x in batch
        )

        self._assert_vision_shapes(y_data, u_data, x_data)

        model_outputs = self.koop_model.forward(
            y_init=y_data[:, 0],
            u_traj=u_data,
            num_steps=num_steps,
        )

        z_proj = self.koop_model.batch_projection(y_data, u_data)

        delay = self.model_params["auto_encoder"].get("delay", 1)
        targets = build_ground_truth_from_images(
            y_data=y_data,
            x_data=x_data,
            drone_dim=self.model_params["drone_dim"],
            delay=delay,
        )

        val_outputs = VisionValForwardOutputs(
            rec=model_outputs.rec,
            pred=model_outputs.pred,
            g_t=targets,
            inputs_scaled=u_data,
            state=x_data,
        )

        # Pour l’instant, VisionLossComputer attend directement model_outputs,
        # z_proj et targets. On garde val_outputs prêt pour une future
        # harmonisation avec SensorValForwardOutputs.
        _ = val_outputs

        assert self.curriculum is not None

        return self.loss_computer.compute(
            model_outputs=model_outputs,
            z_proj=z_proj,
            targets=targets,
            phases_active=self.curriculum.phases_active,
            effective_weight=self._vision_weight_fn,
            num_views=self.model_params["num_views"],
        )

    def reduce_sub_losses(self, sub_losses_arr: list[SubLosses]) -> SubLosses:
        return mean_sub_losses(sub_losses_arr)

    def _vision_weight_fn(self, base: float, key: str) -> float:
        assert self.curriculum is not None
        return self.curriculum.effective_weight(base, key, self.current_epoch)

    def _log_training_state(self, mean_grad_norm: float) -> None:
        if self.modality == "sensor":
            self.logger.log_training_state(
                epoch=self.current_epoch,
                optimizer=self.optimizer,
                total_norm=mean_grad_norm,
            )
        elif self.modality == "vision":
            assert self.curriculum is not None

            # même logique que dans l'ancien VisionTrainer :
            # activation du curriculum à la fin de l'epoch
            self.curriculum.maybe_activate_phases(self.current_epoch)

            self.logger.log_training_state(
                epoch=self.current_epoch,
                optimizer=self.optimizer,
                total_norm=mean_grad_norm,
                base=self.training_params["loss_weights"],
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

    @staticmethod
    def _assert_vision_shapes(
        y_data: Tensor,
        u_data: Tensor,
        x_data: Tensor,
    ) -> None:
        if y_data.dim() != 5:
            raise ValueError(
                "Expected y_data with shape (B,T,C,H,W), "
                f"got {tuple(y_data.shape)}"
            )

        if u_data.dim() != 3:
            raise ValueError(
                "Expected u_data with shape (B,T,n_u), "
                f"got {tuple(u_data.shape)}"
            )

        if x_data.dim() < 2:
            raise ValueError(
                "Expected x_data with at least 2 dimensions, "
                f"got {tuple(x_data.shape)}"
            )

        if y_data.shape[0] != u_data.shape[0]:
            raise ValueError(
                f"Batch size mismatch: y_data={tuple(y_data.shape)}, "
                f"u_data={tuple(u_data.shape)}"
            )

        if y_data.shape[1] != u_data.shape[1]:
            raise ValueError(
                f"Time dimension mismatch: y_data={tuple(y_data.shape)}, "
                f"u_data={tuple(u_data.shape)}"
            )