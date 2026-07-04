from __future__ import annotations

import numpy as np

from KoNAMIC.core.simulation import ClosedLoopTrajectory
from KoNAMIC.pipelines.experiment_comparison import TrajectoryComparisonResult


def closed_loop_trajectory_to_comparison_result(
    trajectory: ClosedLoopTrajectory,
    *,
    dt: float,
) -> TrajectoryComparisonResult:
    if trajectory.x_data is None:
        raise ValueError("Closed-loop trajectory is missing x_data.")

    time = np.asarray(trajectory.time)
    reference_state = np.asarray(trajectory.x_data.ref_traj)
    compared_state = np.asarray(trajectory.x_data.traj)
    inputs = np.asarray(trajectory.inputs_data.u_physical)

    horizon = min(time.shape[0], reference_state.shape[0], compared_state.shape[0])
    if horizon <= 0:
        raise ValueError(
            "Cannot build closed-loop comparison result from empty trajectory."
        )

    input_horizon = min(max(horizon - 1, 0), inputs.shape[0])

    return TrajectoryComparisonResult(
        time=time[:horizon],
        reference_state=reference_state[:horizon],
        compared_state=compared_state[:horizon],
        input_time=time[:input_horizon],
        inputs=inputs[:input_horizon],
        dt=float(dt),
    )
