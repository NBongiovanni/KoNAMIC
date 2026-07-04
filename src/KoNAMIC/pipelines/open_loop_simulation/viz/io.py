from __future__ import annotations

from pathlib import Path

import numpy as np

from KoNAMIC.pipelines.open_loop_simulation.trajectories import (
    saved_rollout_arrays_to_comparison_result,
)
from KoNAMIC.pipelines.experiment_comparison import TrajectoryComparisonResult


def load_rollout_arrays(path: Path) -> dict[str, np.ndarray | float]:
    """
    Load post-processed open-loop rollout arrays saved by render_open_loop_rollouts.

    The experiment_comparison visualizer only needs physical ground truth states, predicted
    states, and inputs. Keeping this loader array-based avoids coupling overlay
    plots back to model output containers or scalers.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "results.npz"

    if not path.exists():
        raise FileNotFoundError(f"Open-loop rollout results not found: {path}")

    with np.load(path) as data:
        required = {"time", "reference_state", "compared_state", "input_time", "inputs"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Missing keys in {path}: {sorted(missing)}")

        rollout = {key: data[key] for key in required}
        if "dt" in data.files:
            rollout["dt"] = float(data["dt"])

        return rollout


def load_open_loop_comparison_result(
    path: Path,
) -> TrajectoryComparisonResult:
    rollout = load_rollout_arrays(path)

    if "dt" not in rollout:
        raise ValueError(
            f"dt is missing from {Path(path) / 'results.npz' if Path(path).is_dir() else path}. "
            "Re-run open-loop rendering to write dt into results.npz."
        )

    dt = float(rollout["dt"])
    return saved_rollout_arrays_to_comparison_result(
        rollout,
        dt=dt,
    )
