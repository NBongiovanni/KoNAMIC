from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.nn import Module


class CheckpointManager:
    def __init__(
        self,
        checkpoints_dir: Path,
        checkpoint_every: int,
    ) -> None:
        self.checkpoints_dir = checkpoints_dir
        self.checkpoint_every = checkpoint_every

    def should_save(self, epoch: int) -> bool:
        return epoch % self.checkpoint_every == 0

    def save(
        self,
        *,
        epoch: int,
        model: Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        path = self.checkpoints_dir / f"model_epoch_{epoch}.pt"

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }

        if extra is not None:
            checkpoint.update(extra)

        torch.save(checkpoint, path)

        return path