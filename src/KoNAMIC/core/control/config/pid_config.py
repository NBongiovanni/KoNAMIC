from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from KoNAMIC.config.config_utils import require_keys

from .base import ControllerConfig


@dataclass(frozen=True, kw_only=True)
class PidControllerConfig(ControllerConfig):
    tracked_state_indices: list[int]
    kp_pos: list[float]
    ki_pos: list[float]
    kd_pos: list[float]
    att_cmd_alpha: float
    thrust_min: float
    thrust_max: float
    deriv_filter_N: float


@dataclass(frozen=True, kw_only=True)
class Quadrotor2DPidConfig(PidControllerConfig):
    kp_att: float
    ki_att: float
    kd_att: float
    moment_max: float
    max_moment_rates: float
    acc_xy_max: float
    phi_ref_max: float | None = None
    theta_ref_max: float | None = None

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "Quadrotor2DPidConfig":
        require_keys(
            cfg,
            [
                "system_name",
                "controller_type",
                "dt",
                "modality",
                "tracked_state_indices",
                "kp_pos",
                "ki_pos",
                "kd_pos",
                "kp_att",
                "ki_att",
                "kd_att",
                "att_cmd_alpha",
                "moment_max",
                "max_moment_rates",
                "acc_xy_max",
                "thrust_min",
                "thrust_max",
                "deriv_filter_N",
            ],
            "controller.pid.quadrotor_2d",
        )
        return cls(
            system_name=str(cfg["system_name"]),
            controller_type=str(cfg["controller_type"]),
            dt=float(cfg["dt"]),
            modality=str(cfg["modality"]),
            tracked_state_indices=[int(v) for v in cfg["tracked_state_indices"]],
            kp_pos=[float(v) for v in cfg["kp_pos"]],
            ki_pos=[float(v) for v in cfg["ki_pos"]],
            kd_pos=[float(v) for v in cfg["kd_pos"]],
            att_cmd_alpha=float(cfg["att_cmd_alpha"]),
            thrust_min=float(cfg["thrust_min"]),
            thrust_max=float(cfg["thrust_max"]),
            deriv_filter_N=float(cfg["deriv_filter_N"]),
            kp_att=float(cfg["kp_att"]),
            ki_att=float(cfg["ki_att"]),
            kd_att=float(cfg["kd_att"]),
            moment_max=float(cfg["moment_max"]),
            max_moment_rates=float(cfg["max_moment_rates"]),
            acc_xy_max=float(cfg["acc_xy_max"]),
            phi_ref_max=(
                float(cfg["phi_ref_max"]) if cfg.get("phi_ref_max") is not None else None
            ),
            theta_ref_max=(
                float(cfg["theta_ref_max"]) if cfg.get("theta_ref_max") is not None else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        data = _pid_common_to_dict(self)
        data.update(
            {
                "kp_att": self.kp_att,
                "ki_att": self.ki_att,
                "kd_att": self.kd_att,
                "moment_max": self.moment_max,
                "max_moment_rates": self.max_moment_rates,
                "acc_xy_max": self.acc_xy_max,
            }
        )
        if self.phi_ref_max is not None:
            data["phi_ref_max"] = self.phi_ref_max
        if self.theta_ref_max is not None:
            data["theta_ref_max"] = self.theta_ref_max
        return data


@dataclass(frozen=True, kw_only=True)
class Quadrotor3DPidConfig(PidControllerConfig):
    kp_att: list[float]
    ki_att: list[float]
    kd_att: list[float]
    moment_max: list[float]
    max_moment_rates: list[float]
    acc_xy_max: float
    phi_ref_max: float
    theta_ref_max: float
    x_init: dict[str, Any] | None = None
    x_ref: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "Quadrotor3DPidConfig":
        require_keys(
            cfg,
            [
                "system_name",
                "controller_type",
                "dt",
                "modality",
                "tracked_state_indices",
                "kp_pos",
                "ki_pos",
                "kd_pos",
                "kp_att",
                "ki_att",
                "kd_att",
                "att_cmd_alpha",
                "moment_max",
                "max_moment_rates",
                "acc_xy_max",
                "thrust_min",
                "thrust_max",
                "deriv_filter_N",
                "phi_ref_max",
                "theta_ref_max",
            ],
            "controller.pid.quadrotor_3d",
        )
        return cls(
            system_name=str(cfg["system_name"]),
            controller_type=str(cfg["controller_type"]),
            dt=float(cfg["dt"]),
            modality=str(cfg["modality"]),
            tracked_state_indices=[int(v) for v in cfg["tracked_state_indices"]],
            kp_pos=[float(v) for v in cfg["kp_pos"]],
            ki_pos=[float(v) for v in cfg["ki_pos"]],
            kd_pos=[float(v) for v in cfg["kd_pos"]],
            att_cmd_alpha=float(cfg["att_cmd_alpha"]),
            thrust_min=float(cfg["thrust_min"]),
            thrust_max=float(cfg["thrust_max"]),
            deriv_filter_N=float(cfg["deriv_filter_N"]),
            kp_att=[float(v) for v in cfg["kp_att"]],
            ki_att=[float(v) for v in cfg["ki_att"]],
            kd_att=[float(v) for v in cfg["kd_att"]],
            moment_max=[float(v) for v in cfg["moment_max"]],
            max_moment_rates=[float(v) for v in cfg["max_moment_rates"]],
            acc_xy_max=float(cfg["acc_xy_max"]),
            phi_ref_max=float(cfg["phi_ref_max"]),
            theta_ref_max=float(cfg["theta_ref_max"]),
            x_init=dict(cfg["x_init"]) if cfg.get("x_init") is not None else None,
            x_ref=dict(cfg["x_ref"]) if cfg.get("x_ref") is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        data = _pid_common_to_dict(self)
        data.update(
            {
                "kp_att": self.kp_att,
                "ki_att": self.ki_att,
                "kd_att": self.kd_att,
                "moment_max": self.moment_max,
                "max_moment_rates": self.max_moment_rates,
                "acc_xy_max": self.acc_xy_max,
                "phi_ref_max": self.phi_ref_max,
                "theta_ref_max": self.theta_ref_max,
            }
        )
        if self.x_init is not None:
            data["x_init"] = self.x_init
        if self.x_ref is not None:
            data["x_ref"] = self.x_ref
        return data


def _pid_common_to_dict(cfg: PidControllerConfig) -> dict[str, Any]:
    return {
        "system_name": cfg.system_name,
        "controller_type": cfg.controller_type,
        "dt": cfg.dt,
        "modality": cfg.modality,
        "tracked_state_indices": cfg.tracked_state_indices,
        "kp_pos": cfg.kp_pos,
        "ki_pos": cfg.ki_pos,
        "kd_pos": cfg.kd_pos,
        "att_cmd_alpha": cfg.att_cmd_alpha,
        "thrust_min": cfg.thrust_min,
        "thrust_max": cfg.thrust_max,
        "deriv_filter_N": cfg.deriv_filter_N,
    }
