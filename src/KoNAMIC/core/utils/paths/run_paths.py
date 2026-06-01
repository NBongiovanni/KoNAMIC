from dataclasses import dataclass
from pathlib import Path

from .path_utils import build_base_output_dir, make_timestamp


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    log_dir: Path

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def eval_dir(self) -> Path:
        return self.run_dir / "eval"

    def training_eval_dir(self, eval_type: str, epoch: int) -> Path:
        """
        Directory for evaluations performed automatically during training.

        Example:
            run_dir/eval/during_training/open_loop/epoch_0020
            run_dir/eval/during_training/closed_loop/epoch_0040
        """
        return (
            self.eval_dir
            / "during_training"
            / eval_type
            / f"epoch_{epoch:04d}"
        )

    def standalone_eval_dir(self, eval_type: str, stamp_eval: str) -> Path:
        """
        Directory for evaluations launched independently from the training loop.

        Example:
            run_dir/eval/standalone/open_loop/2026-05-31_15-20-10
            run_dir/eval/standalone/closed_loop/2026-05-31_15-45-33
        """
        return (
            self.eval_dir
            / "standalone"
            / eval_type
            / stamp_eval
        )


def build_run_paths(
    modality: str,
    drone_dim: int,
    run_status: str,
    stamp_run: str,
) -> RunPaths:

    runs_base_dir = build_base_output_dir(
        modality=modality,
        run_status=run_status,
        drone_dim=drone_dim,
    )

    run_dir = runs_base_dir / "models" / stamp_run
    log_dir = runs_base_dir / "logs" / stamp_run

    run_paths = RunPaths(
        run_dir=run_dir,
        log_dir=log_dir,
    )

    run_paths.run_dir.mkdir(parents=True, exist_ok=True)
    run_paths.log_dir.mkdir(parents=True, exist_ok=True)
    run_paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    return run_paths

def create_run_stamp(dynamics: str, run_id: str, logger) -> str:
    prefix_by_dynamics = {"linear": "lin", "bilinear": "bilin"}

    try:
        prefix = prefix_by_dynamics[dynamics]
    except KeyError as exc:
        raise ValueError(f"Unknown dynamics model: {dynamics}") from exc
    return f"{prefix}_{run_id}_{make_timestamp(logger)}"