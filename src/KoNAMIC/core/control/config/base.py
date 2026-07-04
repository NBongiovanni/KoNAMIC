from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from KoNAMIC.config.config_utils import require_keys


@dataclass(frozen=True, kw_only=True)
class ControllerConfig:
    system_name: str
    controller_type: str
    dt: float
    modality: str | None = None

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "ControllerConfig":
        require_keys(
            cfg,
            ["system_name", "controller_type", "dt"],
            "controller",
        )
        return cls(
            system_name=str(cfg["system_name"]),
            controller_type=str(cfg["controller_type"]),
            dt=float(cfg["dt"]),
            modality=str(cfg["modality"]) if cfg.get("modality") is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "system_name": self.system_name,
            "controller_type": self.controller_type,
            "dt": self.dt,
        }
        if self.modality is not None:
            data["modality"] = self.modality
        return data

    def with_dt(self, dt: float) -> "ControllerConfig":
        return replace(self, dt=dt)
