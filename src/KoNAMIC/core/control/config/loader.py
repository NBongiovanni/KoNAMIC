from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

from KoNAMIC.config.config_utils import load_yaml
from KoNAMIC.paths import find_project_root

from .base import ControllerConfig
from .kmpc_config import KmpcControllerConfig
from .klqr_config import KlqrControllerConfig
from .lqr_config import LqrControllerConfig
from .pid_config import Quadrotor2DPidConfig, Quadrotor3DPidConfig


ControllerConfigT: TypeAlias = (
    KmpcControllerConfig
    | KlqrControllerConfig
    | LqrControllerConfig
    | Quadrotor2DPidConfig
    | Quadrotor3DPidConfig
)


def load_controller_config_from_dict(cfg: dict[str, Any]) -> ControllerConfigT:
    base = ControllerConfig.from_dict(cfg)
    controller_type = _normalize_controller_type(base.controller_type)

    if controller_type == "kmpc":
        return KmpcControllerConfig.from_dict(cfg)

    if controller_type == "klqr":
        return KlqrControllerConfig.from_dict(cfg)

    if controller_type == "lqr":
        return LqrControllerConfig.from_dict(cfg)

    if controller_type == "pid":
        if base.system_name == "quadrotor_2d":
            return Quadrotor2DPidConfig.from_dict(cfg)
        if base.system_name == "quadrotor_3d":
            return Quadrotor3DPidConfig.from_dict(cfg)
        raise ValueError(
            f"Unsupported pid controller system_name={base.system_name!r}."
        )

    raise ValueError(f"Unsupported controller_type={base.controller_type!r}.")


def load_controller_config_from_path(path: Path) -> ControllerConfigT:
    return load_controller_config_from_dict(load_yaml(path))


def load_controller_config(
    controller: str,
    system_name: str,
    modality: str,
    variant: str,
) -> ControllerConfigT:
    project_root = find_project_root()
    controller_dir = (
        project_root
        / "configs"
        / "components"
        / "controllers"
        / controller
    )

    variant = _normalize_controller_variant(
        controller=controller,
        system_name=system_name,
        modality=modality,
        variant=variant,
    )

    candidate_paths = []
    if modality is not None and variant is not None:
        candidate_paths.append(controller_dir / system_name / modality / f"{variant}.yaml")
    if modality is not None:
        candidate_paths.append(controller_dir / system_name / f"{modality}.yaml")
    candidate_paths.append(controller_dir / f"{system_name}.yaml")

    for path in candidate_paths:
        if path.exists():
            return load_controller_config_from_path(path)

    raise FileNotFoundError(
        f"No controller config found for controller={controller!r}, "
        f"system_name={system_name!r}, modality={modality!r}, "
        f"variant={variant!r}."
    )


def _normalize_controller_variant(
    *,
    controller: str,
    system_name: str,
    modality: str | None,
    variant: str | None,
) -> str | None:
    if (
        controller == "kmpc"
        and system_name == "quadrotor_2d"
        and modality == "sensor"
        and variant == "state_in_z"
    ):
        return "position_in_z"
    return variant


def _normalize_controller_type(controller_type: str) -> str:
    if controller_type == "knmpc":
        return "kmpc"
    if controller_type == "koopman_lqr":
        return "klqr"
    return controller_type
