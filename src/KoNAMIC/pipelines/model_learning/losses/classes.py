from dataclasses import dataclass
from typing import TypeAlias

from KoNAMIC.core.simulation import ClosedLoopTrajectory, ClosedLoopMetrics

SubLosses: TypeAlias = dict[str, float]

@dataclass
class OpenLoopLosses:
    full_loss: float
    sub_losses: dict[str, float]


@dataclass
class EpochEvalResult:
    open_loop_losses: dict[str, OpenLoopLosses]
    closed_loop_trajectories: list[ClosedLoopTrajectory] | None = None
    closed_loop_metrics: ClosedLoopMetrics | None = None

    @property
    def val_1(self) -> OpenLoopLosses:
        return self.open_loop_losses["val_1"]

    @property
    def val_2(self) -> OpenLoopLosses:
        return self.open_loop_losses["val_2"]

    def to_phase_losses_dict(self) -> dict[str, OpenLoopLosses]:
        return dict(self.open_loop_losses)
