from dataclasses import dataclass
import numpy as np

from .trajectories import ClosedLoopTrajectory


@dataclass
class ClosedLoopMetrics:
    x_rmse: float
    position_rmse: float
    z_rmse: float
    u_rms: float


def rmse(error: np.ndarray | None) -> float:
    if error is None or error.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(error**2)))


def compute_closed_loop_metrics(results: list[ClosedLoopTrajectory]) -> ClosedLoopMetrics:
    x_errors = [
        r.x_data.error
        for r in results
        if r.x_data is not None and r.x_data.error is not None
    ]

    z_errors = [
        r.z_data.error
        for r in results
        if r.z_data is not None and r.z_data.error is not None
    ]

    u_trajs = [
        r.inputs_data.u_physical
        for r in results
        if r.inputs_data.u_physical.size > 0
    ]

    x_error = np.concatenate(x_errors, axis=0) if x_errors else None
    z_error = np.concatenate(z_errors, axis=0) if z_errors else None
    u_traj = np.concatenate(u_trajs, axis=0) if u_trajs else None

    position_error = x_error[:, :3] if x_error is not None and x_error.shape[1] >= 3 else None

    return ClosedLoopMetrics(
        x_rmse=rmse(x_error),
        position_rmse=rmse(position_error),
        z_rmse=rmse(z_error),
        u_rms=rmse(u_traj),
    )
