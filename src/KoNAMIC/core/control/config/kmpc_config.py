from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeAlias

from KoNAMIC.config.config_utils import load_yaml, require_keys
from KoNAMIC.paths import find_project_root

from .base import ControllerConfig


@dataclass(frozen=True, kw_only=True)
class InputConstraintsConfig:
    use_inputs_constraints: bool
    force_limits: list[float]
    torque_limits: list[float] | None = None

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "InputConstraintsConfig":
        require_keys(
            cfg,
            ["use_inputs_constraints", "force_limits"],
            "controller.constraints",
        )
        return cls(
            use_inputs_constraints=bool(cfg["use_inputs_constraints"]),
            force_limits=[float(v) for v in cfg["force_limits"]],
            torque_limits=(
                [float(v) for v in cfg["torque_limits"]]
                if cfg.get("torque_limits") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "use_inputs_constraints": self.use_inputs_constraints,
            "force_limits": self.force_limits,
        }
        if self.torque_limits is not None:
            data["torque_limits"] = self.torque_limits
        return data


@dataclass(frozen=True, kw_only=True)
class SqpRtiSolverOptionsConfig:
    nlp_solver_type: str
    qp_solver_iter_max: int
    qp_tol: float
    qp_warm_start: bool
    regularization: float
    print_level: int

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "SqpRtiSolverOptionsConfig":
        require_keys(
            cfg,
            [
                "nlp_solver_type",
                "qp_solver_iter_max",
                "qp_tol",
                "qp_warm_start",
                "regularization",
                "print_level",
            ],
            "controller.solver_options",
        )
        _require_solver_type(cfg, expected="SQP_RTI")
        return cls(
            nlp_solver_type=str(cfg["nlp_solver_type"]),
            qp_solver_iter_max=int(cfg["qp_solver_iter_max"]),
            qp_tol=float(cfg["qp_tol"]),
            qp_warm_start=bool(cfg["qp_warm_start"]),
            regularization=float(cfg["regularization"]),
            print_level=int(cfg["print_level"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "nlp_solver_type": self.nlp_solver_type,
            "qp_solver_iter_max": self.qp_solver_iter_max,
            "qp_tol": self.qp_tol,
            "qp_warm_start": self.qp_warm_start,
            "regularization": self.regularization,
            "print_level": self.print_level,
        }


@dataclass(frozen=True, kw_only=True)
class SqpSolverOptionsConfig:
    nlp_solver_type: str
    nlp_solver_iter_max: int
    nlp_tol: float
    qp_solver_iter_max: int
    qp_tol: float
    accept_suboptimal: bool
    regularization: float
    print_level: int

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "SqpSolverOptionsConfig":
        require_keys(
            cfg,
            [
                "nlp_solver_type",
                "nlp_solver_iter_max",
                "nlp_tol",
                "qp_solver_iter_max",
                "qp_tol",
                "accept_suboptimal",
                "regularization",
                "print_level",
            ],
            "controller.solver_options",
        )
        _require_solver_type(cfg, expected="SQP")
        return cls(
            nlp_solver_type=str(cfg["nlp_solver_type"]),
            nlp_solver_iter_max=int(cfg["nlp_solver_iter_max"]),
            nlp_tol=float(cfg["nlp_tol"]),
            qp_solver_iter_max=int(cfg["qp_solver_iter_max"]),
            qp_tol=float(cfg["qp_tol"]),
            accept_suboptimal=bool(cfg["accept_suboptimal"]),
            regularization=float(cfg["regularization"]),
            print_level=int(cfg["print_level"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "nlp_solver_type": self.nlp_solver_type,
            "nlp_solver_iter_max": self.nlp_solver_iter_max,
            "nlp_tol": self.nlp_tol,
            "qp_solver_iter_max": self.qp_solver_iter_max,
            "qp_tol": self.qp_tol,
            "accept_suboptimal": self.accept_suboptimal,
            "regularization": self.regularization,
            "print_level": self.print_level,
        }


SolverOptionsConfig: TypeAlias = SqpRtiSolverOptionsConfig | SqpSolverOptionsConfig

SOLVER_OPTIONS_PROFILE_ALIASES = {
    "linear_latent": "linear_latent_medium",
    "bilinear_latent": "bilinear_latent_medium",
}


@dataclass(frozen=True, kw_only=True)
class KmpcCostConfig:
    mode: str
    R: list[float]
    S: list[float]
    Q_state: float | None = None
    P_state: float | None = None
    Q_positions: float | None = None
    Q_velocities: float | None = None
    P_positions: float | None = None
    P_velocities: float | None = None
    Qz: float | None = None
    Pz: float | None = None
    Q_other_state: float | None = None
    P_other_state: float | None = None
    Q_latent: float | None = None
    P_latent: float | None = None
    only_position: bool | None = None

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "KmpcCostConfig":
        require_keys(cfg, ["mode", "R", "S"], "controller.cost")
        mode = str(cfg["mode"])
        _validate_cost_mode(mode)
        return cls(
            mode=mode,
            R=[float(v) for v in cfg["R"]],
            S=[float(v) for v in cfg["S"]],
            Q_state=float(cfg["Q_state"]) if cfg.get("Q_state") is not None else None,
            P_state=float(cfg["P_state"]) if cfg.get("P_state") is not None else None,
            Q_positions=(
                float(cfg["Q_positions"]) if cfg.get("Q_positions") is not None else None
            ),
            Q_velocities=(
                float(cfg["Q_velocities"]) if cfg.get("Q_velocities") is not None else None
            ),
            P_positions=(
                float(cfg["P_positions"]) if cfg.get("P_positions") is not None else None
            ),
            P_velocities=(
                float(cfg["P_velocities"]) if cfg.get("P_velocities") is not None else None
            ),
            Qz=float(cfg["Qz"]) if cfg.get("Qz") is not None else None,
            Pz=float(cfg["Pz"]) if cfg.get("Pz") is not None else None,
            Q_other_state=(
                float(cfg["Q_other_state"]) if cfg.get("Q_other_state") is not None else None
            ),
            P_other_state=(
                float(cfg["P_other_state"]) if cfg.get("P_other_state") is not None else None
            ),
            Q_latent=float(cfg["Q_latent"]) if cfg.get("Q_latent") is not None else None,
            P_latent=float(cfg["P_latent"]) if cfg.get("P_latent") is not None else None,
            only_position=(
                bool(cfg["only_position"]) if cfg.get("only_position") is not None else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "mode": self.mode,
            "R": self.R,
            "S": self.S,
        }
        optional_fields = {
            "Q_state": self.Q_state,
            "P_state": self.P_state,
            "Q_positions": self.Q_positions,
            "Q_velocities": self.Q_velocities,
            "P_positions": self.P_positions,
            "P_velocities": self.P_velocities,
            "Qz": self.Qz,
            "Pz": self.Pz,
            "Q_other_state": self.Q_other_state,
            "P_other_state": self.P_other_state,
            "Q_latent": self.Q_latent,
            "P_latent": self.P_latent,
            "only_position": self.only_position,
        }
        for key, value in optional_fields.items():
            if value is not None:
                data[key] = value
        return data


@dataclass(frozen=True, kw_only=True)
class KmpcControllerConfig(ControllerConfig):
    constraints: InputConstraintsConfig
    cost: KmpcCostConfig
    solver_options: SolverOptionsConfig
    solver_options_profile: str | None
    num_steps_horizon: int
    controller_type: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "KmpcControllerConfig":
        require_keys(
            cfg,
            ["system_name", "controller_type", "dt", "constraints", "cost", "num_steps_horizon"],
            "controller.kmpc",
        )
        solver_options_profile = (
            str(cfg["solver_options_profile"])
            if cfg.get("solver_options_profile") is not None
            else None
        )
        solver_options = _load_solver_options(
            cfg,
            solver_options_profile=solver_options_profile,
        )
        return cls(
            system_name=str(cfg["system_name"]),
            controller_type=_normalize_kmpc_controller_type(str(cfg["controller_type"])),
            dt=float(cfg["dt"]),
            modality=str(cfg["modality"]) if cfg.get("modality") is not None else None,
            constraints=InputConstraintsConfig.from_dict(cfg["constraints"]),
            cost=KmpcCostConfig.from_dict(cfg["cost"]),
            solver_options=solver_options,
            solver_options_profile=solver_options_profile,
            num_steps_horizon=int(cfg["num_steps_horizon"]),
        )

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "num_steps_horizon": self.num_steps_horizon,
                "constraints": self.constraints.to_dict(),
                "cost": self.cost.to_dict(),
            }
        )
        if self.solver_options_profile is not None:
            data["solver_options_profile"] = self.solver_options_profile
        else:
            data["solver_options"] = self.solver_options.to_dict()
        return data

    def with_solver_options_profile(self, profile: str) -> "KmpcControllerConfig":
        return replace(
            self,
            solver_options_profile=profile,
            solver_options=_load_solver_options_profile(profile),
        )


def _load_solver_options(
    cfg: dict[str, Any],
    *,
    solver_options_profile: str | None,
) -> SolverOptionsConfig:
    if solver_options_profile is not None:
        if cfg.get("solver_options") is not None:
            raise ValueError(
                "controller.kmpc must define either solver_options_profile or "
                "solver_options, not both."
            )
        return _load_solver_options_profile(solver_options_profile)

    if cfg.get("solver_options") is None:
        raise KeyError(
            "Missing required key in controller.kmpc: 'solver_options_profile'. "
            "Legacy inline 'solver_options' is still accepted."
        )
    return _solver_options_from_dict(cfg["solver_options"])


def _load_solver_options_profile(profile: str) -> SolverOptionsConfig:
    profile = SOLVER_OPTIONS_PROFILE_ALIASES.get(profile, profile)
    path = _solver_options_profile_path(profile)
    if not path.exists():
        raise FileNotFoundError(
            f"Unknown KMPC solver_options_profile={profile!r}. "
            f"Expected file: {path}"
        )
    return _solver_options_from_dict(load_yaml(path))


def _solver_options_profile_path(profile: str) -> Path:
    return (
        find_project_root()
        / "configs"
        / "components"
        / "controllers"
        / "kmpc"
        / "solver_options"
        / f"{profile}.yaml"
    )


def _solver_options_from_dict(cfg: dict[str, Any]) -> SolverOptionsConfig:
    require_keys(cfg, ["nlp_solver_type"], "controller.solver_options")
    nlp_solver_type = str(cfg["nlp_solver_type"])
    if nlp_solver_type == "SQP_RTI":
        return SqpRtiSolverOptionsConfig.from_dict(cfg)
    if nlp_solver_type == "SQP":
        return SqpSolverOptionsConfig.from_dict(cfg)
    raise ValueError(
        f"Unsupported controller.solver_options.nlp_solver_type={nlp_solver_type!r}."
    )


def _require_solver_type(cfg: dict[str, Any], *, expected: str) -> None:
    actual = str(cfg["nlp_solver_type"])
    if actual != expected:
        raise ValueError(
            f"Invalid solver options type: expected nlp_solver_type={expected!r}, "
            f"got {actual!r}."
        )


def _normalize_kmpc_controller_type(controller_type: str) -> str:
    if controller_type == "knmpc":
        return "kmpc"
    return controller_type


def _validate_cost_mode(mode: str) -> None:
    valid_modes = {"state_in_z", "position_in_z", "full_latent", "structured_latent"}
    if mode not in valid_modes:
        raise ValueError(
            f"Unsupported controller.cost.mode={mode!r}. "
            f"Expected one of {sorted(valid_modes)}."
        )
