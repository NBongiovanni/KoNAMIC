from __future__ import annotations

from typing import Optional

import numpy as np

from .data import InputsData, TrajectoryData
from .trajectories import ClosedLoopTrajectory


def build_closed_loop_trajectory(
    *,
    time: np.ndarray,
    x_init: np.ndarray,
    x_traj: np.ndarray,
    x_ref_traj: Optional[np.ndarray] = None,
    u_physical: Optional[np.ndarray] = None,
    u_scaled: Optional[np.ndarray] = None,
    z_traj: Optional[np.ndarray] = None,
    z_ref_traj: Optional[np.ndarray] = None,
    im_traj: Optional[np.ndarray] = None,
    im_ref_traj: Optional[np.ndarray] = None,
) -> ClosedLoopTrajectory:

    x_init = np.asarray(x_init, dtype=float).reshape(-1)
    time = np.asarray(time, dtype=float).reshape(-1)

    x_traj = np.asarray(x_traj, dtype=float)
    x_ref_traj = _coerce_reference_like_traj(x_ref_traj, x_traj)
    x_error = _compute_error_if_possible(x_traj, x_ref_traj)

    x_data = TrajectoryData(
        traj=x_traj,
        ref_traj=x_ref_traj,
        error=x_error,
    )

    z_traj = _ensure_2d_or_empty(z_traj)
    z_ref_traj = _coerce_reference_like_traj(z_ref_traj, z_traj)
    z_error = _compute_error_if_possible(z_traj, z_ref_traj)

    z_data = TrajectoryData(
        traj=z_traj,
        ref_traj=z_ref_traj,
        error=z_error,
    )

    im_data = TrajectoryData(
        traj=im_traj,
        ref_traj=im_ref_traj,
        error=None,
    )

    if u_physical is None:
        u_physical = np.zeros((0, 0), dtype=float)
    else:
        u_physical = np.asarray(u_physical, dtype=float)

    if u_scaled is None:
        u_scaled = u_physical.copy()
    else:
        u_scaled = np.asarray(u_scaled, dtype=float)

    inputs_data = InputsData(
        u_physical=u_physical,
        u_scaled=u_scaled,
    )

    return ClosedLoopTrajectory(
        time=time,
        x_init=x_init,
        x_data=x_data,
        z_data=z_data,
        im_data=im_data,
        inputs_data=inputs_data,
    )


def _ensure_2d_or_empty(arr: Optional[np.ndarray]) -> np.ndarray:
    if arr is None:
        return np.zeros((0, 0), dtype=float)

    arr = np.asarray(arr, dtype=float)

    if arr.ndim == 1:
        return arr[:, None]

    return arr


def _coerce_reference_like_traj(
    ref: Optional[np.ndarray],
    traj: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if ref is None or traj is None:
        return ref

    ref = np.asarray(ref, dtype=float)
    traj = np.asarray(traj, dtype=float)

    if ref.ndim == 1 and traj.ndim == 2 and ref.shape[0] == traj.shape[1]:
        ref = np.repeat(ref[None, :], traj.shape[0], axis=0)

    if ref.ndim == 2 and traj.ndim == 2 and ref.shape[0] != traj.shape[0]:
        n = min(ref.shape[0], traj.shape[0])
        ref = ref[:n]

    return ref


def _compute_error_if_possible(
    traj: Optional[np.ndarray],
    ref: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if traj is None or ref is None:
        return None

    traj = np.asarray(traj, dtype=float)
    ref = np.asarray(ref, dtype=float)

    if traj.ndim == 1:
        traj = traj[:, None]

    if ref.ndim == 1 and traj.ndim == 2 and ref.shape[0] == traj.shape[1]:
        ref = np.repeat(ref[None, :], traj.shape[0], axis=0)

    if traj.shape != ref.shape:
        n = min(traj.shape[0], ref.shape[0])
        traj = traj[:n]
        ref = ref[:n]

    if traj.shape != ref.shape:
        return None

    return traj - ref