from __future__ import annotations

from pathlib import Path
from typing import Union
import yaml

from KoNAMIC import paths
from .drone.drone_spec import DroneSpec
from .cartpole.cartpole_spec import CartPoleSpec


SystemSpec = Union[DroneSpec, CartPoleSpec]


def create_system(system_name: str) -> SystemSpec:
    project_root = paths.find_project_root()
    config_path = project_root / "configs" / "components" / "systems" / f"{system_name}.yaml"

    system_type = _read_system_type(config_path)

    if system_type == "quadrotor":
        return DroneSpec.from_yaml(config_path)

    if system_type == "cartpole":
        return CartPoleSpec.from_yaml(config_path)

    raise ValueError(
        f"Unsupported system_type={system_type!r} in {config_path}. "
        "Expected one of: 'quadrotor', 'cartpole'."
    )


def _read_system_type(config_path: Path) -> str:
    if not config_path.exists():
        raise FileNotFoundError(f"System config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected system config to contain a mapping/dict, "
            f"got {type(data).__name__} in {config_path}"
        )

    if "system_type" not in data:
        raise KeyError(
            f"Missing required key 'system_type' in {config_path}. "
            "Example: system_type: quadrotor"
        )

    return str(data["system_type"])