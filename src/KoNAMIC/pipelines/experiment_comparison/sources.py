from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from KoNAMIC.config.cases_loader import CaseConfig
from KoNAMIC.paths.open_loop_paths import build_rollout_results_dir

if TYPE_CHECKING:
    from KoNAMIC.config.presets_loader.open_loop import OpenLoopComparisonConfig
    from KoNAMIC.config.viz_config import ClosedLoopComparisonConfig


@dataclass(frozen=True)
class ComparisonSource:
    label: str
    color: str
    results_dir: Path
    run_status: str
    kind: str = "experiment"


def build_open_loop_sources(
    *,
    cfg: OpenLoopComparisonConfig,
    cases: Sequence[CaseConfig],
    system_dim: int,
) -> list[ComparisonSource]:
    _validate_source_inputs(cfg=cfg, cases=cases)

    sources: list[ComparisonSource] = []

    for model_cfg, case in zip(cfg.models, cases):
        rollout_0_dir = build_rollout_results_dir(
            cfg=cfg,
            system_dim=system_dim,
            case=case,
            rollout_idx=0,
        )
        sources.append(
            ComparisonSource(
                label=model_cfg.label,
                color=model_cfg.color,
                results_dir=rollout_0_dir.parent,
                run_status=case.run_status,
                kind="experiment",
            )
        )

    return sources


def build_closed_loop_sources(
    *,
    cfg: ClosedLoopComparisonConfig,
    cases: Sequence[CaseConfig],
) -> tuple[list[ComparisonSource], int]:
    _validate_source_inputs(cfg=cfg, cases=cases)

    sources: list[ComparisonSource] = []
    system_dim = cases[0].system_dim

    for model_cfg, case in zip(cfg.models, cases):
        if case.system_dim != system_dim:
            raise ValueError(
                f"All compared experiments must share the same system_dim. "
                f"Got {system_dim} and {case.system_dim}."
            )

        sources.append(
            ComparisonSource(
                label=model_cfg.label,
                color=model_cfg.color,
                results_dir=build_closed_loop_results_dir(
                    cfg=cfg,
                    case=case,
                    trajectory_type=cfg.trajectory_type,
                ),
                run_status=case.run_status,
                kind="experiment",
            )
        )

    return sources, system_dim


def build_closed_loop_results_dir(
    *,
    cfg: ClosedLoopComparisonConfig,
    case: CaseConfig,
    trajectory_type: str,
) -> Path:
    if cfg.task not in case.best_simulations:
        available_tasks = ", ".join(sorted(case.best_simulations.keys()))
        raise KeyError(
            f"Task {cfg.task!r} not found for model {case.model_stamp!r}. "
            f"Available tasks: {available_tasks}."
        )

    task_simulations = case.best_simulations[cfg.task]
    if trajectory_type not in task_simulations:
        available_trajectories = ", ".join(sorted(task_simulations.keys()))
        raise KeyError(
            f"Trajectory type {trajectory_type!r} not found for "
            f"task {cfg.task!r} and model {case.model_stamp!r}. "
            f"Available trajectory types: {available_trajectories}."
        )

    simulation_stamp = task_simulations[trajectory_type]

    return (
        cfg.output_dir
        / case.run_status
        / cfg.modality
        / cfg.system_name
        / "runs"
        / case.stamp_run
        / "eval"
        / "standalone"
        / cfg.task
        / simulation_stamp
    )


def _validate_source_inputs(*, cfg, cases: Sequence[CaseConfig]) -> None:
    if not cfg.models:
        raise ValueError("cfg.models must contain at least one model.")
    if len(cases) != len(cfg.models):
        raise ValueError(
            f"Expected one case per model, got {len(cases)} cases "
            f"for {len(cfg.models)} models."
        )
