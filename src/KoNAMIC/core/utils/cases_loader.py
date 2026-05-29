from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

from .paths.path_utils import find_project_root


@dataclass(frozen=True)
class CaseConfig:
    model_stamp: str
    epoch: int
    run_status: str
    drone_dim: int
    best_simulations: dict[str, dict[str, str]]


_CASES_FILENAMES: Final[dict[str, str]] = {
    "vision": "vision.yaml",
    "sensor": "sensor.yaml",
}


def _get_cases_file(modality: str) -> Path:
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
    return (
        project_root
        / "configs"
        / "registries"
        / "models"
        / filename
    )


def _parse_case_config(case_id: str, case_dict: Any, path: Path) -> CaseConfig:
    """
    Parse and validate one model registry entry.
    """
    if not isinstance(case_dict, dict):
        raise TypeError(
            f"Invalid case {case_id!r} in {path}: expected a mapping, "
            f"got {type(case_dict).__name__}."
        )

    required_keys = {
        "model_stamp",
        "epoch",
        "run_status",
        "drone_dim",
        "best_simulations",
    }

    missing_keys = required_keys - case_dict.keys()
    if missing_keys:
        raise KeyError(
            f"Invalid case {case_id!r} in {path}: missing required key(s) "
            f"{sorted(missing_keys)}."
        )

    best_simulations = case_dict["best_simulations"]
    if not isinstance(best_simulations, dict):
        raise TypeError(
            f"Invalid case {case_id!r} in {path}: 'best_simulations' "
            f"must be a mapping."
        )

    drone_dim = int(case_dict["drone_dim"])
    if drone_dim not in (1, 2, 3):
        raise ValueError(
            f"Invalid case {case_id!r} in {path}: "
            f"'drone_dim' must be 1, 2, or 3, got {drone_dim}."
        )

    return CaseConfig(
        model_stamp=str(case_dict["model_stamp"]),
        epoch=int(case_dict["epoch"]),
        run_status=str(case_dict["run_status"]),
        drone_dim=drone_dim,
        best_simulations=best_simulations,
    )


def load_cases(modality: str) -> dict[int, CaseConfig]:
    """
    Load the model registry associated with a modality.

    Parameters
    ----------
    modality:
        Model modality. Expected values: "vision" or "sensor".

    Returns
    -------
    dict[int, CaseConfig]
        Dictionary indexed by case identifier.
    """
    path = _get_cases_file(modality)

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
        int(case_id): _parse_case_config(case_id, case_dict, path)
        for case_id, case_dict in raw_cases.items()
    }