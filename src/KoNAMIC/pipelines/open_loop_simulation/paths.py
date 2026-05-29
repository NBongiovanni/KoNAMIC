from pathlib import Path

from KoNAMIC.core.utils import CaseConfig
from .configs import OpenLoopComparisonConfig


def build_comparison_dir(cfg: OpenLoopComparisonConfig, drone_dim: int) -> Path:
    return (
        cfg.output_dir
        / cfg.run_status
        / cfg.modality
        / f"{drone_dim}d"
        / "figures"
        / cfg.task
        / cfg.comparison_name
    )


def get_best_simulation_stamp(
    *,
    cfg: OpenLoopComparisonConfig,
    case: CaseConfig,
) -> str:
    """
    Return the simulation stamp associated with the requested task
    and trajectory type.

    Expected structure:

    best_simulations:
      open_loop:
        setpoint_tracking: "..."
        trajectory_tracking: "..."
      closed_loop:
        setpoint_tracking: "..."
        trajectory_tracking: "..."
    """
    if cfg.task not in case.best_simulations:
        available_tasks = ", ".join(sorted(case.best_simulations.keys()))
        raise KeyError(
            f"Task {cfg.task!r} not found for model {case.model_stamp!r}. "
            f"Available tasks: {available_tasks}."
        )

    task_simulations = case.best_simulations[cfg.task]

    if cfg.trajectory_type not in task_simulations:
        available_trajectories = ", ".join(sorted(task_simulations.keys()))
        raise KeyError(
            f"Trajectory type {cfg.trajectory_type!r} not found for "
            f"task {cfg.task!r} and model {case.model_stamp!r}. "
            f"Available trajectory types: {available_trajectories}."
        )

    return task_simulations[cfg.trajectory_type]


def build_rollout_results_dir(
    cfg: OpenLoopComparisonConfig,
    drone_dim: int,
    case: CaseConfig,
    rollout_idx: int,
) -> Path:
    stamp_simulation = get_best_simulation_stamp(
        cfg=cfg,
        case=case,
    )

    return (
        cfg.output_dir
        / cfg.run_status
        / cfg.modality
        / f"{drone_dim}d"
        / "models"
        / case.model_stamp
        / "eval"
        / cfg.task
        / stamp_simulation
        / f"rollout_{rollout_idx}"
    )