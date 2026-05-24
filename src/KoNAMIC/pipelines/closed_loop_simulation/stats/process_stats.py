from __future__ import annotations

import numpy as np

from .simulation_statistics import SimulationStatistics


def _safe_array(arr):
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return None
    return arr


def _safe_mean_abs(arr) -> float:
    arr = _safe_array(arr)
    if arr is None:
        return 0.0
    return float(np.mean(np.abs(arr)))


def _safe_terminal_norm(arr) -> float:
    arr = _safe_array(arr)
    if arr is None:
        return 0.0
    if arr.ndim == 1:
        return float(np.linalg.norm(arr))
    return float(np.linalg.norm(arr[-1]))


def _compute_overshoot_and_settling_time(
    x_traj: np.ndarray | None,
    x_ref_traj: np.ndarray | None,
    dt: float,
    state_idx: int,
    tol_ratio: float = 0.05,
) -> tuple[float, float]:
    """
    Overshoot en % et settling time en secondes pour une composante scalaire.
    Retourne (0.0, 0.0) si non calculable.
    """
    if x_traj is None or x_ref_traj is None:
        return 0.0, 0.0

    x_traj = np.asarray(x_traj, dtype=float)
    x_ref_traj = np.asarray(x_ref_traj, dtype=float)

    if x_traj.ndim != 2 or x_ref_traj.ndim != 2:
        return 0.0, 0.0

    n = min(len(x_traj), len(x_ref_traj))
    if n == 0:
        return 0.0, 0.0

    y = x_traj[:n, state_idx]
    r = x_ref_traj[:n, state_idx]

    r_final = r[-1]
    y0 = y[0]
    step_amp = r_final - y0

    if abs(step_amp) < 1e-12:
        return 0.0, 0.0

    if step_amp > 0:
        peak = np.max(y)
        overshoot = max(0.0, (peak - r_final) / abs(step_amp) * 100.0)
    else:
        trough = np.min(y)
        overshoot = max(0.0, (r_final - trough) / abs(step_amp) * 100.0)

    band = tol_ratio * abs(step_amp)
    error = np.abs(y - r)

    settling_idx = None
    for k in range(n):
        if np.all(error[k:] <= band):
            settling_idx = k
            break

    settling_time = 0.0 if settling_idx is None else float(settling_idx * dt)
    return float(overshoot), settling_time


def process_statistics(sim_result, sqp_iters: list[int]) -> SimulationStatistics:
    x_data = sim_result.x_data
    z_data = sim_result.z_data

    x_error = None if x_data is None else x_data.error
    x_traj = None if x_data is None else x_data.traj
    x_ref_traj = None if x_data is None else x_data.ref_traj

    z_error = None if z_data is None else z_data.error

    error_x_pos = _safe_mean_abs(x_error)
    error_z = _safe_mean_abs(z_error)
    terminal_error = _safe_terminal_norm(x_error)

    sqp_iter = float(np.mean(sqp_iters)) if len(sqp_iters) > 0 else 0.0

    has_diverged = False
    if x_traj is not None:
        x_traj_arr = np.asarray(x_traj, dtype=float)
        has_diverged = (
            np.any(~np.isfinite(x_traj_arr)) or np.max(np.abs(x_traj_arr)) > 1e3
        )

    overshoot_1, settling_time_1 = _compute_overshoot_and_settling_time(
        x_traj=x_traj,
        x_ref_traj=x_ref_traj,
        dt=sim_result.time[1] - sim_result.time[0] if len(sim_result.time) > 1 else 0.0,
        state_idx=0,
    )

    overshoot_2, settling_time_2 = _compute_overshoot_and_settling_time(
        x_traj=x_traj,
        x_ref_traj=x_ref_traj,
        dt=sim_result.time[1] - sim_result.time[0] if len(sim_result.time) > 1 else 0.0,
        state_idx=1,
    )

    return SimulationStatistics(
        error_x_pos=error_x_pos,
        error_z=error_z,
        terminal_error=terminal_error,
        sqp_iter=sqp_iter,
        has_diverged=has_diverged,
        overshoot_1=overshoot_1,
        overshoot_2=overshoot_2,
        settling_time_1=settling_time_1,
        settling_time_2=settling_time_2,
    )