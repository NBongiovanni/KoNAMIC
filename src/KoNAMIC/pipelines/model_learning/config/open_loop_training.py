from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from KoNAMIC.config.config_utils import require_keys


@dataclass(frozen=True)
class OpenLoopTrainingConfig:
    system_name: str
    modality: str
    enabled: bool
    frequency: int

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "OpenLoopTrainingConfig":
        require_keys(
            cfg,
            ["system_name", "modality", "enabled", "frequency"],
            "open_loop_training",
        )
        return cls(
            system_name=str(cfg["system_name"]),
            modality=str(cfg["modality"]),
            enabled=bool(cfg["enabled"]),
            frequency=int(cfg["frequency"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_name": self.system_name,
            "modality": self.modality,
            "enabled": self.enabled,
            "frequency": self.frequency,
        }
