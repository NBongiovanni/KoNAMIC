from __future__ import annotations

from typing import Literal

from KoNAMIC.core.drone import DroneSpec
from .sensor_generation_config import SensorGenerationConfig

Profile2D = Literal["hover", "step_z", "step_y", "step_yz"]
Profile3D = Literal["hover", "step_z", "step_x", "step_y", "step_xyz"]
Profile = Profile2D | Profile3D


def get_profile(
    cfg: SensorGenerationConfig,
    traj_idx: int,
    num_traj: int,
    drone: DroneSpec,
) -> Profile:

    if drone.drone_dim == 2:
        return get_profile_2d(
            cfg=cfg,
            traj_idx=traj_idx,
            num_traj=num_traj,
        )

    if drone.drone_dim == 3:
        return get_profile_3d(
            cfg=cfg,
            traj_idx=traj_idx,
            num_traj=num_traj,
        )

    raise ValueError(f"Unsupported drone_dim: {drone.drone_dim}")


def get_profile_2d(
    *,
    cfg: SensorGenerationConfig,
    traj_idx: int,
    num_traj: int,
) -> Profile2D:
    if cfg.only_aggressive:
        return "step_yz"

    if traj_idx < num_traj / 10:
        return "hover"
    if traj_idx < 2 * num_traj / 10:
        return "step_z"
    if traj_idx < 3 * num_traj / 10:
        return "step_y"

    return "step_yz"


def get_profile_3d(
    *,
    cfg: SensorGenerationConfig,
    traj_idx: int,
    num_traj: int,
) -> Profile3D:
    if cfg.only_aggressive:
        return "step_xyz"

    if traj_idx < num_traj / 10:
        return "hover"
    if traj_idx < 2 * num_traj / 10:
        return "step_z"
    if traj_idx < 3 * num_traj / 10:
        return "step_x"
    if traj_idx < 4 * num_traj / 10:
        return "step_y"

    return "step_xyz"


def select_one_traj_per_profile(
    *,
    config: SensorGenerationConfig,
    num_traj: int,
    drone: DroneSpec,
) -> tuple[int, ...]:
    traj_indices_by_profile = {}

    for traj_idx in range(num_traj):
        profile = get_profile(
            config,
            traj_idx,
            num_traj,
            drone,
        )

        if profile not in traj_indices_by_profile:
            traj_indices_by_profile[profile] = traj_idx

    return tuple(traj_indices_by_profile.values())