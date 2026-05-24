from dataclasses import dataclass
from pathlib import Path

from KoNAMIC.core import utils
from KoNAMIC.pipelines.closed_loop_simulation import ClosedLoopComparisonConfig


@dataclass(frozen=True)
class SimulationSource:
    label: str
    color: str
    result_path: Path


def build_model_result_path(
    cfg: ClosedLoopComparisonConfig,
    case,
    trajectory_type: str,
) -> Path:
    simulation_stamp = case.closed_loop_simulations[trajectory_type]

    return (
        cfg.output_dir
        / cfg.run_status
        / cfg.modality
        / f"{case.drone_dim}d"
        / "models"
        / case.stamp
        / "eval"
        / "control"
        / simulation_stamp
        / "run_0"
        / "results.pkl"
    )


def build_pid_result_path(
    cfg: ClosedLoopComparisonConfig,
    drone_dim: int,
) -> Path:
    return (
        cfg.output_dir
        / cfg.run_status
        / cfg.modality
        / f"{drone_dim}d"
        / "models"
        / cfg.pid.results_path
    )


def build_simulation_sources(
    cfg: ClosedLoopComparisonConfig,
) -> tuple[list[SimulationSource], int]:
    if not cfg.models:
        raise ValueError("cfg.models must contain at least one model.")

    cases = utils.load_cases(cfg.modality)
    sources: list[SimulationSource] = []
    first_case = cases[cfg.models[0].case_id]
    drone_dim = first_case.drone_dim

    for model_cfg in cfg.models:
        case = cases[model_cfg.case_id]

        if case.drone_dim != drone_dim:
            raise ValueError(
                f"All compared models must share the same drone_dim. "
                f"Got {drone_dim} and {case.drone_dim}."
            )

        result_path = build_model_result_path(cfg, case, cfg.trajectory_type)

        sources.append(
            SimulationSource(
                label=model_cfg.label,
                color=model_cfg.color,
                result_path=result_path,
            )
        )

    if cfg.pid.enabled:
        sources.append(
            SimulationSource(
                label=cfg.pid.label,
                color=cfg.pid.color,
                result_path=build_pid_result_path(cfg, drone_dim),
            )
        )

    return sources, drone_dim