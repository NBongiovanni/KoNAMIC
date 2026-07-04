from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .run_paths import build_run_paths, RunPaths
from .paths_utils import build_baseline_output_dir


@dataclass(frozen=True)
class ClosedLoopRunPaths:
    eval_dir: Path
    koopman_run_dir: Optional[Path] = None


def build_standalone_control_paths(
    *,
    system_name: str,
    controller_type: str,
    modality: str,
    run_status: str,
    stamp_control: str,
) -> RunPaths:
    control_base_dir = build_baseline_output_dir(
        modality=modality,
        run_status=run_status,
        system_name=system_name,
        controller_type=controller_type,
    )

    run_paths = RunPaths(
        run_dir=control_base_dir / "runs" / stamp_control,
        log_dir=control_base_dir / "logs" / stamp_control,
    )

    run_paths.run_dir.mkdir(parents=True, exist_ok=True)
    run_paths.log_dir.mkdir(parents=True, exist_ok=True)

    return run_paths


def build_closed_loop_run_paths(
    *,
    system_name: str,
    controller: str,
    modality: str,
    run_status: str,
    stamp_eval: str,
    stamp_run: str,
) -> ClosedLoopRunPaths:

    if controller in {"kmpc", "klqr"}:
        run_paths = build_run_paths(
            system_name=system_name,
            modality=modality,
            run_status=run_status,
            stamp_run=stamp_run
        )

        return ClosedLoopRunPaths(
            eval_dir=run_paths.standalone_eval_dir(
                eval_type="closed_loop",
                stamp_eval=stamp_eval,
            ),
            koopman_run_dir=run_paths.run_dir,
        )
    else:
        eval_dir = build_standalone_control_paths(
            system_name=system_name,
            controller_type=controller,
            modality="sensor",
            run_status="interim",
            stamp_control=stamp_eval,
        )

        return ClosedLoopRunPaths(
            eval_dir=eval_dir.standalone_eval_dir("closed_loop"),
        )
