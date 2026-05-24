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
    val_1: OpenLoopLosses
    val_2: OpenLoopLosses
    closed_loop_trajectories: list[ClosedLoopTrajectory] | None = None
    closed_loop_metrics: ClosedLoopMetrics | None = None

    def to_phase_losses_dict(self) -> dict[str, tuple[float, SubLosses]]:
        """
        Format compatible avec votre logging existant.
        """
        return {
            "val_1": (self.val_1.full_loss, self.val_1.sub_losses),
            "val_2": (self.val_2.full_loss, self.val_2.sub_losses),
        }