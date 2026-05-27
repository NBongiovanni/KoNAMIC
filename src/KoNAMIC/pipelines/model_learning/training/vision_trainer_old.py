from typing import Dict, Sequence

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter

from KoNAMIC.core.models import VisionKoopModel
from .base_old import BaseTrainer
from ..losses import VisionLossComputer, _mean_sub_losses_vision
from ..loss_logging import log_training_state_vision
from ..curriculum import CurriculumManager
from ..ground_truth import build_ground_truth_from_images


class VisionTrainer(BaseTrainer):
    @property
    def tag(self) -> str:
        return "vision"

    def __init__(
        self,
        training_params: dict,
        koop_model: VisionKoopModel,
        data_loader: dict,
        num_steps_train: int,
        num_steps_val_1: int,
        num_steps_val_2: int,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        writer: SummaryWriter,
        num_views: int,
    ):
        self.num_views = num_views
        self.curriculum = CurriculumManager(training_params["curriculum"])
        self.loss_computer = VisionLossComputer(training_params["loss_weights"])

        super().__init__(
            training_params=training_params,
            koop_model=koop_model,
            data_loader=data_loader,
            num_steps_train=num_steps_train,
            num_steps_val_1=num_steps_val_1,
            num_steps_val_2=num_steps_val_2,
            optimizer=optimizer,
            scheduler=scheduler,
            writer=writer,
            checkpoints_dir=training_params["checkpoints_dir"],
            checkpoints_every=int(training_params["checkpoint_every"]),
            grad_clip_max_norm=float(training_params.get("grad_clip_max_norm", 1.0)),
        )

    def forward_and_loss(
            self,
            batch: Sequence[torch.Tensor],
            num_steps_pred: int,
    ) -> tuple[torch.Tensor, dict]:
        y_data, u_data, x_data = (x.to(self.device, non_blocking=True) for x in batch)

        model_outputs = self.koop_model.forward(
            y_data[:, 0],
            u_data,
            num_steps_pred,
        )
        z_proj = self.koop_model.batch_projection(y_data, u_data)
        targets = build_ground_truth_from_images(
            y_data,
            x_data,
            self.params["drone_dim"],
            self.params["delay"]
        )

        return self.loss_computer.compute(
            model_outputs,
            z_proj,
            targets,
            self.curriculum.phases_active,
            self._weight_fn,
            self.num_views,
        )

    def _weight_fn(self, base: float, key: str) -> float:
        return self.curriculum.effective_weight(base, key, self.current_epoch)

    def reduce_sub_losses(self, sub_losses_arr: Sequence[dict]) -> Dict[str, float]:
        return _mean_sub_losses_vision(sub_losses_arr)

    def log_training_state(self, mean_grad_norm: float) -> None:
        # activation du curriculum à la fin d'epoch, comme dans votre code actuel
        self.curriculum.maybe_activate_phases(self.current_epoch)

        log_training_state_vision(
            self.writer,
            self.optimizer,
            self.params["loss_weights"],
            self._weight_fn,
            mean_grad_norm,
            self.current_epoch,
            self.curriculum.current_phase_index() + 1,
        )