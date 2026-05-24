from pathlib import Path

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


def build_rollout_results_dir(
    cfg: OpenLoopComparisonConfig,
    drone_dim: int,
    case,
    rollout_idx: int,
) -> Path:
    if cfg.trajectory_type not in case.best_open_loop_simulations:
        available = ", ".join(case.best_open_loop_simulations.keys())
        raise KeyError(
            f"Trajectory type '{cfg.trajectory_type}' not found in case '{case.stamp}'. "
            f"Available keys: {available}"
        )

    stamp_simulation = case.best_open_loop_simulations[cfg.trajectory_type]

    return (
        cfg.output_dir
        / cfg.run_status
        / cfg.modality
        / f"{drone_dim}d"
        / "models"
        / case.stamp
        / "eval"
        / cfg.task
        / stamp_simulation
        / f"rollout_{rollout_idx}"
    )