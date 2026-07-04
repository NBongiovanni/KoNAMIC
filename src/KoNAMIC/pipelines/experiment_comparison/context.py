from __future__ import annotations

from argparse import Namespace
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Protocol, TypeVar

from KoNAMIC import config
from KoNAMIC.config import CaseConfig


class ModelComparisonConfig(Protocol):
    case_id: str | int
    label: str
    color: str


class ComparisonConfig(Protocol):
    modality: str
    models: Sequence[ModelComparisonConfig]


ConfigT = TypeVar("ConfigT", bound=ComparisonConfig)


@dataclass(frozen=True)
class ExperimentComparisonContext(Generic[ConfigT]):
    cfg: ConfigT
    cases: list[CaseConfig]
    system_dim: int
    names: list[str]
    colors: list[str]


def build_experiment_comparison_context(
    args: Namespace,
    load_preset: Callable[[Path, str, str, str], ConfigT],
) -> ExperimentComparisonContext[ConfigT]:
    cfg = load_preset(
        args.preset_file,
        args.system_name,
        args.preset,
        args.modality,
    )

    cases_by_id = config.load_cases(cfg.modality, args.system_name)
    selected_cases: list[CaseConfig] = []

    for model_cfg in cfg.models:
        try:
            case = cases_by_id[model_cfg.case_id]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                f"Invalid case_id={model_cfg.case_id!r} for "
                f"modality={cfg.modality!r} and system_name={args.system_name!r}."
            ) from exc
        selected_cases.append(case)

    if not selected_cases:
        raise ValueError("No experiments found in comparison configuration.")

    system_dim = selected_cases[0].system_dim
    for case in selected_cases[1:]:
        if case.system_dim != system_dim:
            raise ValueError("All compared experiments must share the same system_dim.")

    return ExperimentComparisonContext(
        cfg=cfg,
        cases=selected_cases,
        system_dim=system_dim,
        names=[model.label for model in cfg.models],
        colors=[model.color for model in cfg.models],
    )
