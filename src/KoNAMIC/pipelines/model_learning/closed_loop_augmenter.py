from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import DataLoader

from KoNAMIC.core.simulation import ClosedLoopTrajectory
from KoNAMIC.pipelines.model_learning.closed_loop_replay import ClosedLoopReplayBuffer
from .config import ClosedLoopTrainingConfig
from .closed_loop_trajectory_generator import ClosedLoopTrajectoryGenerator


@dataclass
class ClosedLoopAugmenter:
    config: ClosedLoopTrainingConfig
    replay_buffer: ClosedLoopReplayBuffer
    generator: ClosedLoopTrajectoryGenerator

    def is_enabled(self) -> bool:
        return self.config.enabled

    def should_generate(self, epoch: int) -> bool:
        if not self.config.enabled:
            return False

        start_epoch = self.config.start_epoch if self.config.start_epoch is not None else 0
        if epoch < start_epoch:
            return False

        return epoch % self.config.closed_loop_every == 0

    def should_use_replay(self, epoch: int) -> bool:
        if not self.config.enabled:
            return False

        replay_start = self.config.replay_start_epoch_effective
        if replay_start is None:
            return False

        return epoch >= replay_start

    def maybe_generate(self, *, epoch: int) -> list[ClosedLoopTrajectory] | None:
        if not self.should_generate(epoch):
            return None

        trajectories = self.generator.generate(epoch=epoch)
        self.replay_buffer.add(trajectories)
        return trajectories

    def make_dataloader(self, *, epoch: int) -> DataLoader | None:
        if not self.should_use_replay(epoch):
            return None
        return self.replay_buffer.make_dataloader()

    def num_trajectories(self) -> int:
        return len(self.replay_buffer._trajectories)
