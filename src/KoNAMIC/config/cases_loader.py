from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

from KoNAMIC.paths import find_project_root


@dataclass(frozen=True)
class CaseConfig:
    stamp_run: str
    epoch: int
    run_status: str
    system_name: str
    system_dim: int
    best_simulations: dict[str, dict[str, str]]

    @property
    def model_stamp(self) -> str:
        return self.stamp_run

    @property
    def drone_dim(self) -> int:
        return self.system_dim


_CASES_FILENAMES: Final[dict[str, str]] = {
    "vision": "vision.yaml",
    "sensor": "sensor.yaml",
}


def _get_cases_file(modality: str, system_name: str | None = None) -> Path:
    """
    Return the absolute path of the model registry YAML file
    associated with a modality.
    """
    try:
        filename = _CASES_FILENAMES[modality]
    except KeyError as exc:
        valid_modalities = ", ".join(sorted(_CASES_FILENAMES))
        raise ValueError(
            f"Unknown modality {modality!r}. Expected one of: {valid_modalities}."
        ) from exc

    project_root = find_project_root()
    base_path = (
        project_root
        / "configs"
        / "registries"
        / "experiments"
    )

    if system_name is not None:
        return base_path / system_name / filename

    return base_path / filename


def _parse_case_config(case_id: str, case_dict: Any, path: Path) -> CaseConfig:
    """
    Parse and validate one model registry entry.
    """
    if not isinstance(case_dict, dict):
        raise TypeError(
            f"Invalid case {case_id!r} in {path}: expected a mapping, "
            f"got {type(case_dict).__name__}."
        )

    required_keys = {"epoch", "run_status", "best_simulations"}

    missing_keys = required_keys - case_dict.keys()
    if missing_keys:
        raise KeyError(
            f"Invalid case {case_id!r} in {path}: missing required key(s) "
            f"{sorted(missing_keys)}."
        )

    stamp_run = case_dict.get("stamp_run", case_dict.get("model_stamp"))
    if stamp_run is None:
        raise KeyError(
            f"Invalid case {case_id!r} in {path}: missing required key "
            "'stamp_run'."
        )

    system_name = case_dict.get("system_name")
    system_dim_raw = case_dict.get("system_dim", case_dict.get("drone_dim"))

    if system_name is None and system_dim_raw is None:
        raise KeyError(
            f"Invalid case {case_id!r} in {path}: missing required key "
            "'system_name'."
        )

    if system_dim_raw is None:
        system_dim = _infer_system_dim(str(system_name))
    else:
        system_dim = int(system_dim_raw)

    best_simulations = case_dict["best_simulations"]
    if not isinstance(best_simulations, dict):
        raise TypeError(
            f"Invalid case {case_id!r} in {path}: 'best_simulations' "
            f"must be a mapping."
        )

    if system_dim not in (1, 2, 3):
        raise ValueError(
            f"Invalid case {case_id!r} in {path}: "
            f"'system_dim' must be 1, 2, or 3, got {system_dim}."
        )

    return CaseConfig(
        stamp_run=str(stamp_run),
        epoch=int(case_dict["epoch"]),
        run_status=str(case_dict["run_status"]),
        system_name=str(system_name) if system_name is not None else f"{system_dim}d",
        system_dim=system_dim,
        best_simulations=best_simulations,
    )


def load_cases(modality: str, system_name: str | None = None) -> dict[str | int, CaseConfig]:
    """
    Load the model registry associated with a modality.

    Parameters
    ----------
    modality:
        Model modality. Expected values: "vision" or "sensor".
    system_name

    Returns
    -------
    dict[str | int, CaseConfig]
        Dictionary indexed by case identifier.
    """
    path = _get_cases_file(modality, system_name)

    if not path.exists():
        raise FileNotFoundError(f"Model registry file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise TypeError(
            f"Invalid YAML structure in {path}: expected a mapping at root."
        )

    if "model_registry" not in data:
        raise KeyError(f"Missing top-level key 'model_registry' in {path}")

    raw_cases = data["model_registry"]
    if not isinstance(raw_cases, dict):
        raise TypeError(
            f"Invalid 'model_registry' section in {path}: expected a mapping."
        )

    return {
        _parse_case_id(case_id): _parse_case_config(case_id, case_dict, path)
        for case_id, case_dict in raw_cases.items()
    }


def _parse_case_id(case_id: str) -> str | int:
    try:
        return int(case_id)
    except ValueError:
        return case_id


def _infer_system_dim(system_name: str) -> int:
    if system_name.endswith("_2d"):
        return 2
    if system_name.endswith("_3d"):
        return 3
    if system_name == "cartpole":
        return 1

    raise ValueError(
        f"Cannot infer system_dim from system_name={system_name!r}. "
        "Add 'system_dim' to the case registry entry."
    )
