from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from KoNAMIC.config.config_utils import require_keys

from .base import ControllerConfig
from .kmpc_config import InputConstraintsConfig


@dataclass(frozen=True, kw_only=True)
class KlqrCostConfig:
    q_diag: list[float]
    r_diag: list[float]

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "KlqrCostConfig":
        require_keys(cfg, ["q_diag", "r_diag"], "controller.klqr.cost")
        return cls(
            q_diag=[float(v) for v in cfg["q_diag"]],
            r_diag=[float(v) for v in cfg["r_diag"]],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "q_diag": self.q_diag,
            "r_diag": self.r_diag,
        }


@dataclass(frozen=True, kw_only=True)
class KlqrControllerConfig(ControllerConfig):
    constraints: InputConstraintsConfig
    cost: KlqrCostConfig

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "KlqrControllerConfig":
        require_keys(
            cfg,
            ["system_name", "controller_type", "dt", "modality", "constraints", "cost"],
            "controller.klqr",
        )
        return cls(
            system_name=str(cfg["system_name"]),
            controller_type=_normalize_klqr_controller_type(str(cfg["controller_type"])),
            dt=float(cfg["dt"]),
            modality=str(cfg["modality"]),
            constraints=InputConstraintsConfig.from_dict(cfg["constraints"]),
            cost=KlqrCostConfig.from_dict(cfg["cost"]),
        )

    @property
    def q_diag(self) -> list[float]:
        return self.cost.q_diag

    @property
    def r_diag(self) -> list[float]:
        return self.cost.r_diag

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "constraints": self.constraints.to_dict(),
                "cost": self.cost.to_dict(),
            }
        )
        return data


def _normalize_klqr_controller_type(controller_type: str) -> str:
    if controller_type == "koopman_lqr":
        return "klqr"
    return controller_type
