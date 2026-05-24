from dataclasses import dataclass
from typing import Optional, Sequence

@dataclass
class RampConfig:
    """
    Ramp schedule configuration.

    Attributes:
        start_epoch (Optional[int]): Epoch index where the ramp starts;
            None means "not yet started".
        duration (int): Number of epochs to ramp from 0 to 1
            (clamped to [0, 1] during evaluation).
    """
    start_epoch: Optional[int]
    duration: int

@dataclass
class CurriculumCfg:
    """
    Configuration du curriculum d'entraînement.
    """
    epoch_triggers: Sequence[Optional[int]]
    ramp_c: RampConfig
    ramp_a: RampConfig


class CurriculumManager:
    """
    Gère l'activation des phases de curriculum et les poids effectifs de certaines pertes.
    """
    def __init__(self, curriculum_params: dict):
        self.cfg = CurriculumCfg(
            epoch_triggers=tuple(curriculum_params["phase_epoch_triggers"]),
            ramp_c=RampConfig(None, curriculum_params["ramp_duration"]),
            ramp_a=RampConfig(None, curriculum_params["ramp_duration"]),
        )

        self.phases_active = [False] * len(self.cfg.epoch_triggers)

    def maybe_activate_phases(self, epoch: int) -> None:
        """
        Active les phases dont l'epoch de déclenchement est atteint.
        """
        for idx, trigger_epoch in enumerate(self.cfg.epoch_triggers):
            if self.phases_active[idx]: # Phase déjà active -> on passe
                continue

            if epoch >= trigger_epoch:
                self._activate_phase(idx, epoch)

    def _activate_phase(self, idx: int, epoch: int) -> None:
        """Active la phase d'indice idx et initialise les rampes si nécessaire."""
        self.phases_active[idx] = True
        if idx == 0:
            self.cfg.ramp_c.start_epoch = epoch
            self.cfg.ramp_a.start_epoch = epoch

    def current_phase_index(self) -> int:
        """
        Retourne l'indice de la phase active la plus avancée.
        """
        active_indices = [i for i, active in enumerate(self.phases_active) if active]
        return max(active_indices, default=-1)

    def effective_weight(self, base: float, key: str, epoch: int) -> float:
        """
        Calcule le poids effectif d'un terme de loss ("c" ou "a") pour un epoch donné.
        """
        if not self.phases_active[0]:
            return 0.0
        ramp_cfg = self.cfg.ramp_c if key == "c" else self.cfg.ramp_a
        ramp_factor = _linear_ramp(epoch, ramp_cfg.start_epoch, ramp_cfg.duration)
        return base * ramp_factor


def _linear_ramp(t: int, t0: int, duration: int) -> float:
    """
    Linear ramp from 0 to 1 over `duration` epochs starting at `t0`.

    Args:
        t (int): Current epoch.
        t0 (int): Start epoch (None means not started yet).
        duration (int): Ramp length in epochs. Non-positive duration yields 1.0.

    Returns:
        float: Ramp factor in [0, 1].

    Notes:
        - When `t0 is None`, returns 0.0.
        - Uses a clamped linear rule: clamp((t − t0 + 1) / duration, 0, 1).
          Adjust the +1 offset if you prefer the first active epoch to be 0.0.
    """
    if t0 is None:
        return 0.0
    if duration <= 0:
        return 1.0
    return max(0.0, min(1.0, (t - t0 + 1) / duration))
