from __future__ import annotations

from typing import Literal

from .params import SensorDatasetParams

Profile2D = Literal["hover", "step_z", "step_y", "step_yz"]
Profile3D = Literal["hover", "step_z", "step_x", "step_y", "step_xyz"]

Profile = Profile2D | Profile3D


def get_profile(
    cfg: SensorDatasetParams,
    traj_idx: int,
    drone,
) -> Profile:
    if drone.drone_dim == 2:
        return get_profile_2d(cfg=cfg, traj_idx=traj_idx)

    if drone.drone_dim == 3:
        return get_profile_3d(cfg=cfg, traj_idx=traj_idx)

    raise ValueError(f"Unsupported drone_dim: {drone.drone_dim}")


def get_profile_2d(
    *,
    cfg: SensorDatasetParams,
    traj_idx: int,
) -> Profile2D:
    if cfg.only_aggressive:
        return "step_yz"

    n = cfg.num_traj

    if traj_idx < n / 10:
        return "hover"
    if traj_idx < 2 * n / 10:
        return "step_z"
    if traj_idx < 3 * n / 10:
        return "step_y"

    return "step_yz"


def get_profile_3d(
    *,
    cfg: SensorDatasetParams,
    traj_idx: int,
) -> Profile3D:
    if cfg.only_aggressive:
        return "step_xyz"

    n = cfg.num_traj

    if traj_idx < n / 10:
        return "hover"
    if traj_idx < 2 * n / 10:
        return "step_z"
    if traj_idx < 3 * n / 10:
        return "step_x"
    if traj_idx < 4 * n / 10:
        return "step_y"

    return "step_xyz"