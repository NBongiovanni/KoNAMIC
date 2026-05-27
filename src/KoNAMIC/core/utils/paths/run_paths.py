from dataclasses import dataclass
from pathlib import Path

from .path_utils import build_base_output_dir, make_timestamp


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    log_dir: Path
    open_loop_eval_dir: Path | None = None
    closed_loop_eval_dir: Path | None = None

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"


def build_run_paths(
    modality: str,
    drone_dim: int,
    run_status: str,
    stamp_run: str,
    stamp_open_loop: str | None = None,
    stamp_closed_loop: str | None = None,
) -> RunPaths:

    runs_base_dir = build_base_output_dir(
        modality=modality,
        run_status=run_status,
        drone_dim=drone_dim,
    )
    run_dir = runs_base_dir / "models" / stamp_run
    log_dir = runs_base_dir / "logs" / stamp_run
    checkpoints_dir = run_dir / stamp_run

    if stamp_open_loop is not None:
        open_loop_eval_dir = run_dir / "eval" / "open_loop" / stamp_open_loop
    else:
        open_loop_eval_dir = None

    if stamp_closed_loop is not None:
        closed_loop_eval_dir = run_dir / "eval" / "closed_loop" / stamp_closed_loop
    else:
        closed_loop_eval_dir = None

    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        run_dir=run_dir,
        log_dir=log_dir,
        open_loop_eval_dir=open_loop_eval_dir,
        closed_loop_eval_dir=closed_loop_eval_dir,
    )


def create_run_stamp(dynamics: str, run_id: str, logger) -> str:
    prefix_by_dynamics = {
        "linear": "lin",
        "bilinear": "bilin",
    }

    try:
        prefix = prefix_by_dynamics[dynamics]
    except KeyError as exc:
        raise ValueError(f"Unknown dynamics model: {dynamics}") from exc

    return f"{prefix}_{run_id}_{make_timestamp(logger)}"