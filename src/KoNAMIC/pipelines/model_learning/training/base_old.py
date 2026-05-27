from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from KSIC_v8.models import BaseKoopModel
from ..loss_logging import log_losses_console, log_losses_tensorboard

class BaseTrainer(ABC):
    """
    Minimal common trainer (template method) for SensorTrainer and VisionTrainer.

    Subclasses must implement:
      - forward_and_loss(batch, num_steps_pred) -> (full_loss_tensor, sub_losses_dict)
      - reduce_sub_losses(list_of_sub_losses_dict) -> dict
      - log_training_state(mean_grad_norm) -> None
      - tag property ("sensor" / "vision")
    """

    def __init__(
        self,
        training_params: dict,
        koop_model: BaseKoopModel,
        data_loader: dict,
        num_steps_train: int,
        num_steps_val_1: int,
        num_steps_val_2: int,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        writer: SummaryWriter,
        checkpoints_dir: Path,
        checkpoints_every: int,
        grad_clip_max_norm: float = 1.0,
    ):
        self.params = training_params
        self.koop_model: BaseKoopModel = koop_model
        self.data_loader = data_loader

        self.num_steps_train = num_steps_train
        self.num_steps_val_1 = num_steps_val_1
        self.num_steps_val_2 = num_steps_val_2

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = writer

        self.device = next(self.koop_model.parameters()).device
        self.checkpoints_dir = checkpoints_dir
        self.checkpoints_every = checkpoints_every
        self.num_val_datasets = int(training_params.get("num_val_datasets", 2))

        self.grad_clip_max_norm = float(grad_clip_max_norm)
        self.current_epoch = 0

    # ---- required hooks -------------------------------------------------
    @property
    @abstractmethod
    def tag(self) -> str:
        """Used for logging ('sensor' or 'vision')."""

    @abstractmethod
    def forward_and_loss(
        self,
        batch: Sequence[torch.Tensor],
        num_steps_pred: int,
    ) -> tuple[torch.Tensor, dict]:
        """Return (full_loss_tensor, sub_losses_dict)."""

    @abstractmethod
    def reduce_sub_losses(self, sub_losses_arr: Sequence[dict]) -> dict:
        """Aggregate dicts of sub-losses into mean values."""

    @abstractmethod
    def log_training_state(self, mean_grad_norm: float) -> None:
        """Log optimizer/LR/curriculum/etc. Called once per epoch."""

    # ---- common training logic -----------------------------------------
    def train_model(self) -> None:
        for epoch in range(self.params["num_epochs"]):
            self.current_epoch = epoch
            phase_losses = {
                "train": self.train_one_epoch(),
                "val_1": self.eval_model("val_1", self.num_steps_val_1),
                "val_2": self.eval_model("val_2", self.num_steps_val_2),
            }
            self.log_losses(phase_losses)
            if epoch % self.checkpoints_every == 0:
                self.save_model()
        self.writer.flush()
        self.writer.close()

    def train_one_epoch(self) -> Tuple[float, Dict[str, float]]:
        self.koop_model.train()
        full_loss_arr, sub_losses_arr, total_norm_arr = [], [], []

        for batch in tqdm(self.data_loader["train"]):
            self.optimizer.zero_grad(set_to_none=True)
            full_loss, sub_losses = self.forward_and_loss(batch, self.num_steps_train)
            full_loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.koop_model.parameters(),
                self.grad_clip_max_norm,
            )
            self.optimizer.step()

            full_loss_arr.append(float(full_loss.detach().cpu().item()))
            sub_losses_arr.append(sub_losses)
            total_norm_arr.append(float(total_norm.detach().cpu().item()))

        self.scheduler.step()

        full_mean = float(np.mean(full_loss_arr)) if full_loss_arr else float("nan")
        sub_mean = self.reduce_sub_losses(sub_losses_arr)
        mean_grad_norm = float(np.mean(total_norm_arr)) if total_norm_arr else float("nan")

        self.log_training_state(mean_grad_norm)
        return full_mean, sub_mean

    @torch.no_grad()
    def eval_model(self, phase: str, num_steps_val: int) -> Tuple[float, Dict[str, float]]:
        self.koop_model.eval()
        full_loss_arr, sub_losses_arr = [], []

        for batch in tqdm(self.data_loader[phase]):
            full_loss, sub_losses = self.forward_and_loss(batch, num_steps_val)
            full_loss_arr.append(float(full_loss.cpu().item()))
            sub_losses_arr.append(sub_losses)

        full_mean = float(np.mean(full_loss_arr)) if full_loss_arr else float("nan")
        sub_mean = self.reduce_sub_losses(sub_losses_arr)
        return full_mean, sub_mean

    def save_model(self) -> None:
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoints_dir / f"model_epoch_{self.current_epoch}.pt"
        torch.save(
            {
                "epoch": self.current_epoch,
                "model_state_dict": self.koop_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            path,
        )

    def log_losses(self, phase_losses: dict) -> None:
        log_losses_console(
            self.tag,
            self.current_epoch,
            phase_losses,
            self.num_val_datasets,
        )
        log_losses_tensorboard(
            self.tag,
            self.writer,
            self.current_epoch,
            phase_losses,
            self.num_val_datasets,
        )