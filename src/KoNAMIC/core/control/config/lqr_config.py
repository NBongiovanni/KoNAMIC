from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from KoNAMIC.config.config_utils import require_keys

from .base import ControllerConfig


@dataclass(frozen=True, kw_only=True)
class LqrControllerConfig(ControllerConfig):
    q_diag: list[float]
    r_diag: list[float]
    force_min: float | None = None
    force_max: float | None = None
    thrust_min: float | None = None
    thrust_max: float | None = None
    moment_max: float | None = None
    max_moment_rates: float | None = None

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "LqrControllerConfig":
        require_keys(
            cfg,
            ["system_name", "controller_type", "dt", "q_diag", "r_diag"],
            "controller.lqr",
        )
        return cls(
            system_name=str(cfg["system_name"]),
            controller_type=str(cfg["controller_type"]),
            dt=float(cfg["dt"]),
            modality=str(cfg["modality"]) if cfg.get("modality") is not None else None,
            q_diag=[float(v) for v in cfg["q_diag"]],
            r_diag=[float(v) for v in cfg["r_diag"]],
            force_min=float(cfg["force_min"]) if cfg.get("force_min") is not None else None,
            force_max=float(cfg["force_max"]) if cfg.get("force_max") is not None else None,
            thrust_min=(
                float(cfg["thrust_min"]) if cfg.get("thrust_min") is not None else None
            ),
            thrust_max=(
                float(cfg["thrust_max"]) if cfg.get("thrust_max") is not None else None
            ),
            moment_max=(
                float(cfg["moment_max"]) if cfg.get("moment_max") is not None else None
            ),
            max_moment_rates=(
                float(cfg["max_moment_rates"])
                if cfg.get("max_moment_rates") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["q_diag"] = self.q_diag
        data["r_diag"] = self.r_diag
        optional_fields = {
            "force_min": self.force_min,
            "force_max": self.force_max,
            "thrust_min": self.thrust_min,
            "thrust_max": self.thrust_max,
            "moment_max": self.moment_max,
            "max_moment_rates": self.max_moment_rates,
        }
        for key, value in optional_fields.items():
            if value is not None:
                data[key] = value
        return data
