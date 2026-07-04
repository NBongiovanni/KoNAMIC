from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from KoNAMIC.core.scaling import DatasetScalers
from KoNAMIC.core.simulation import ClosedLoopTrajectory
from .config import ClosedLoopTrainingConfig


BatchType = tuple[Tensor, Tensor]


@dataclass
class ReplayConfig:
    closed_loop_config: ClosedLoopTrainingConfig

    enabled: bool = field(init=False)
    max_num_trajectories: int = field(init=False)
    batch_size: int = field(init=False)

    shuffle: bool = True
    num_workers: int = 0
    drop_last: bool = False

    def __post_init__(self) -> None:
        self.enabled = self.closed_loop_config.enabled
        self.max_num_trajectories = self.closed_loop_config.max_num_trajectories
        self.batch_size = self.closed_loop_config.batch_size


class ClosedLoopReplayBuffer:
    def __init__(
        self,
        scalers: DatasetScalers,
        num_steps: int,
        config: ReplayConfig,
    ) -> None:
        self.scalers = scalers
        self.x_scaler = scalers.x
        self.u_scaler = scalers.u
        self.num_steps = num_steps
        self.config = config
        self._trajectories: deque[ClosedLoopTrajectory] = deque(
            maxlen=config.max_num_trajectories
        )

    def add(self, trajectories: Iterable[ClosedLoopTrajectory] | None) -> None:
        if not self.config.enabled or trajectories is None:
            return

        for traj in trajectories:
            self._trajectories.append(traj)

    def is_empty(self) -> bool:
        return len(self._trajectories) == 0

    def make_dataloader(self) -> DataLoader | None:
        if not self.config.enabled or self.is_empty():
            return None

        dataset = ClosedLoopTrajectoryDataset(
            trajectories=list(self._trajectories),
            x_scaler=self.x_scaler,
            u_scaler=self.u_scaler,
            num_steps=self.num_steps,
        )

        if len(dataset) == 0:
            return None

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            drop_last=self.config.drop_last,
        )


class ClosedLoopTrajectoryDataset(Dataset):
    def __init__(
        self,
        trajectories: list[ClosedLoopTrajectory],
        x_scaler: StandardScaler,
        u_scaler: StandardScaler,
        num_steps: int,
    ) -> None:
        self.samples: list[BatchType] = []

        for traj in trajectories:
            sample = self._trajectory_to_sample(
                traj=traj,
                x_scaler=x_scaler,
                u_scaler=u_scaler,
                num_steps=num_steps,
            )

            if sample is not None:
                self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> BatchType:
        return self.samples[idx]

    @staticmethod
    def _trajectory_to_sample(
        traj: ClosedLoopTrajectory,
        x_scaler: StandardScaler,
        u_scaler: StandardScaler,
        num_steps: int,
    ) -> BatchType | None:
        if traj.x_data is None:
            return None

        x_physical = np.asarray(traj.x_data.traj)
        u_physical = np.asarray(traj.inputs_data.u_physical)

        required_x_len = num_steps
        required_u_len = num_steps

        if x_physical.shape[0] < required_x_len:
            return None

        if u_physical.shape[0] < required_u_len:
            return None

        x_physical = x_physical[:required_x_len]
        u_physical = u_physical[:required_u_len]

        x_scaled = x_scaler.transform(x_physical).astype(np.float32)
        u_scaled = u_scaler.transform(u_physical).astype(np.float32)

        return (
            torch.from_numpy(x_scaled),
            torch.from_numpy(u_scaled),
        )

