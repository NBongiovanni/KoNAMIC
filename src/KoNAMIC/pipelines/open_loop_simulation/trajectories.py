from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np

from KoNAMIC.core.models import SensorValForwardOutputs, VisionValForwardOutputs
from KoNAMIC.pipelines.experiment_comparison import TrajectoryComparisonResult
from KoNAMIC.utils import to_numpy


def sensor_output_to_comparison_result(
    output: SensorValForwardOutputs,
    *,
    dt: float,
    idx_traj: int | None = None,
) -> TrajectoryComparisonResult:
    if output.state_gt_physical is None:
        raise ValueError("Sensor output is missing state_gt_physical.")
    if output.inputs_physical is None:
        raise ValueError("Sensor output is missing inputs_physical.")

    state_gt = _select_trajectory(output.state_gt_physical, idx_traj)
    state_pred = _select_trajectory(output.pred.state, idx_traj)
    inputs = _select_trajectory(output.inputs_physical, idx_traj)

    horizon = min(state_gt.shape[0] - 1, state_pred.shape[0], inputs.shape[0])
    if horizon <= 0:
        raise ValueError(
            "Cannot build sensor open-loop comparison result from empty trajectory."
        )

    return TrajectoryComparisonResult(
        time=np.arange(1, horizon + 1) * dt,
        reference_state=state_gt[1:1 + horizon],
        compared_state=state_pred[:horizon],
        input_time=np.arange(horizon) * dt,
        inputs=inputs[:horizon],
        dt=float(dt),
    )


def vision_output_to_comparison_result(
    output: VisionValForwardOutputs,
    *,
    dt: float,
    idx_traj: int | None = None,
) -> TrajectoryComparisonResult:
    if output.g_t.state is None:
        raise ValueError("Vision output is missing g_t.state.")
    if output.pred.state is None:
        raise ValueError("Vision output is missing pred.state.")
    if output.inputs_physical is None:
        raise ValueError("Vision output is missing inputs_physical.")

    state_gt = _select_trajectory(output.g_t.state, idx_traj)
    state_pred = _select_trajectory(output.pred.state, idx_traj)
    inputs = _select_trajectory(output.inputs_physical, idx_traj)

    horizon = min(state_gt.shape[0], state_pred.shape[0], inputs.shape[0])
    if horizon <= 0:
        raise ValueError(
            "Cannot build vision open-loop comparison result from empty trajectory."
        )

    return TrajectoryComparisonResult(
        time=np.arange(1, horizon + 1) * dt,
        reference_state=state_gt[:horizon],
        compared_state=state_pred[:horizon],
        input_time=np.arange(horizon) * dt,
        inputs=inputs[:horizon],
        dt=float(dt),
    )


def saved_rollout_arrays_to_comparison_result(
    rollout: dict[str, np.ndarray | float],
    *,
    dt: float,
) -> TrajectoryComparisonResult:
    required = {"time", "reference_state", "compared_state", "input_time", "inputs"}
    missing = required - set(rollout)
    if missing:
        raise ValueError(f"Missing rollout arrays: {sorted(missing)}")

    return TrajectoryComparisonResult(
        time=np.asarray(rollout["time"]),
        reference_state=np.asarray(rollout["reference_state"]),
        compared_state=np.asarray(rollout["compared_state"]),
        input_time=np.asarray(rollout["input_time"]),
        inputs=np.asarray(rollout["inputs"]),
        dt=float(dt),
    )


def _select_trajectory(x: Any, idx_traj: int | None) -> np.ndarray:
    x_np = to_numpy(x)
    if idx_traj is not None:
        return np.asarray(x_np[idx_traj])
    if x_np.ndim >= 3:
        if x_np.shape[0] != 1:
            raise ValueError(
                "idx_traj must be provided when converting a batched open-loop output."
            )
        return np.asarray(x_np[0])
    return np.asarray(x_np)

@dataclass
class OpenLoopSensorResult:
    val_output: SensorValForwardOutputs
    u_scaler: Any
    x_scaler: Any
    run_dir: Path
    open_loop_eval_dir: Path
