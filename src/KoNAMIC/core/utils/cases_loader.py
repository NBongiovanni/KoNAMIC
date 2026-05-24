from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import yaml

from .path_utils import find_project_root


@dataclass(frozen=True)
class CaseConfig:
    stamp: str
    epoch: int
    run_status: str
    include_state_in_z: bool
    geom_losses: bool
    drone_dim: int
    dynamics: str
    best_open_loop_simulations: dict
    best_closed_loop_simulations: dict


_CASES_FILENAMES: Final[dict[str, str]] = {
    "vision": "vision.yaml",
    "sensor": "sensor.yaml",
}


def _get_cases_file(modality: str) -> Path:
    """
    Retourne le chemin absolu du fichier YAML correspondant à la modalité.
    """
    try:
        filename = _CASES_FILENAMES[modality]
    except KeyError as exc:
        valid_modalities = ", ".join(sorted(_CASES_FILENAMES))
        raise ValueError(
            f"Unknown modality {modality!r}. Expected one of: {valid_modalities}."
        ) from exc

    project_root = find_project_root()
    return project_root / "configs" / "model_registry" /filename


def load_cases(modality: str) -> dict[int, CaseConfig]:
    """
    Charge les cas de contrôle définis dans le YAML associé à la modalité.

    Parameters
    ----------
    modality:
        Modalité du modèle. Valeurs attendues : "vision" ou "sensor".

    Returns
    -------
    dict[int, CaseConfig]
        Dictionnaire indexé par identifiant de cas.

    Raises
    ------
    ValueError
        Si la modalité est inconnue.
    FileNotFoundError
        Si le fichier YAML n'existe pas.
    KeyError
        Si la clé 'model_registry' est absente du YAML.
    TypeError
        Si un cas ne correspond pas à la structure attendue.
    """
    path = _get_cases_file(modality)

    if not path.exists():
        raise FileNotFoundError(f"Cases config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Invalid YAML structure in {path}: expected a mapping at root.")

    if "model_registry" not in data:
        raise KeyError(f"Missing top-level key 'model_registry' in {path}")

    raw_cases = data["model_registry"]
    if not isinstance(raw_cases, dict):
        raise TypeError(f"Invalid 'model_registry' section in {path}: expected a mapping.")

    try:
        return {
            int(case_id): CaseConfig(**case_dict)
            for case_id, case_dict in raw_cases.items()
        }
    except TypeError as exc:
        raise TypeError(
            f"Invalid case entry in {path}. A case does not match CaseConfig."
        ) from exc