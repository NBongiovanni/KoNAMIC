from pathlib import Path
from typing import Dict, Sequence

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter

from KSIC_v8.models import SensorKoopModel as KoopModel
from KSIC_v8.models.outputs.sensor_outputs import SensorValForwardOutputs

from KSIC_v8.model_learning.trainers.base import BaseTrainer
from KSIC_v8.model_learning.sensor_compute_losses import SensorLossComputer, _mean_sub_losses
from KSIC_v8.model_learning.loss_logging import log_training_state_sensors


class SensorTrainer(BaseTrainer):
    @property
    def tag(self) -> str:
        return "sensor"

    def __init__(
        self,
        training_params: dict,
        koop_model: KoopModel,
        dataloader: dict,
        num_steps_train: int,
        num_steps_val_1: int,
        num_steps_val_2: int,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        writer: SummaryWriter,
        run_dir: Path,
    ):
        self.loss_computer = SensorLossComputer(training_params["loss_weights"])

        super().__init__(
            training_params=training_params,
            koop_model=koop_model,
            data_loader=dataloader,
            num_steps_train=num_steps_train,
            num_steps_val_1=num_steps_val_1,
            num_steps_val_2=num_steps_val_2,
            optimizer=optimizer,
            scheduler=scheduler,
            writer=writer,
            checkpoints_dir=run_dir / "checkpoints",
            checkpoints_every=int(training_params["checkpoint_every"]),
            grad_clip_max_norm=float(training_params.get("grad_clip_max_norm", 1.0)),
        )

    def forward_and_loss(
        self,
        batch: Sequence[torch.Tensor],
        num_steps_pred: int,
    ) -> tuple[torch.Tensor, dict]:
        x_gt_scaled, u_traj_scaled = (x.to(self.device, non_blocking=True) for x in batch)

        rec, pred = self.koop_model.forward(
            x_gt_scaled[:, 0],
            u_traj_scaled,
            num_steps_pred,
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

    def reduce_sub_losses(self, sub_losses_arr: Sequence[dict]) -> Dict[str, float]:
        return _mean_sub_losses(sub_losses_arr)

    def log_training_state(self, mean_grad_norm: float) -> None:
        log_training_state_sensors(
            self.writer,
            self.optimizer,
            mean_grad_norm,
            self.current_epoch,
        )