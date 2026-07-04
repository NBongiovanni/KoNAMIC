from __future__ import annotations

from dataclasses import dataclass

from KoNAMIC.core.simulation import ClosedLoopTrajectory


@dataclass
class KoopmanLQRAugmentationRunner:
    # system_spec, plant, scenario_generator, koop_model, scalers, etc.
    # plus tard: contrôleur Koopman LQR
    def generate(self, *, epoch: int) -> list[ClosedLoopTrajectory]:
        # lancer les rollouts closed loop
        # retourner les trajectoires
        raise NotImplementedError