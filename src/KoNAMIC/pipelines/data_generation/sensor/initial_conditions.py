from __future__ import annotations

import numpy as np

from .sensor_generation_config import SensorGenerationConfig
from .profile import Profile
from .references import rand_uniform_sym


def sample_initial_condition(
    *,
    cfg: SensorGenerationConfig,
    profile: Profile,
    rng: np.random.Generator,
    drone,
) -> np.ndarray:
    angle_max = 0.0 if cfg.init_angles_to_zero else cfg.angle_init_max

    if drone.drone_dim == 2:
        return sample_initial_condition_2d(
            cfg=cfg,
            profile=profile,
            rng=rng,
            angle_max=angle_max,
            drone=drone,
        )

    if drone.drone_dim == 3:
        return sample_initial_condition_3d(
            cfg=cfg,
            profile=profile,
            rng=rng,
            angle_max=angle_max,
            drone=drone,
        )

    raise ValueError(f"Unsupported drone_dim: {drone.drone_dim}")


def sample_initial_condition_2d(
    *,
    cfg: SensorGenerationConfig,
    profile: Profile,
    rng: np.random.Generator,
    angle_max: float,
    drone,
) -> np.ndarray:
    bounds = {
        "hover": (0.0, 0.0, 0.0),
        "step_z": (0.0, cfg.z_init_max, 0.0),
        "step_y": (cfg.y_init_max, 0.0, 0.0),
        "step_yz": (cfg.y_init_max, cfg.z_init_max, angle_max),
    }
    y_max, z_max, theta_max = bounds[profile]

    x0 = np.zeros(drone.x_dim, dtype=float)
    x0[0] = rand_uniform_sym(rng, y_max)
    x0[1] = rand_uniform_sym(rng, z_max)
    x0[2] = rand_uniform_sym(rng, theta_max)

    return x0


def sample_initial_condition_3d(
    *,
    cfg: SensorGenerationConfig,
    profile: Profile,
    rng: np.random.Generator,
    angle_max: float,
    drone,
) -> np.ndarray:
    bounds = {
        "hover":    (0.0, 0.0, 0.0, 0.0,      0.0),
        "step_z":   (0.0, 0.0, cfg.z_init_max, 0.0,      0.0),
        "step_x":   (cfg.x_init_max, 0.0, 0.0, 0.0,      0.0),
        "step_y":   (0.0, cfg.y_init_max, 0.0, 0.0,      0.0),
        "step_xyz": (cfg.x_init_max, cfg.y_init_max, cfg.z_init_max, angle_max, angle_max),
    }

    x_max, y_max, z_max, phi_max, theta_max = bounds[profile]

    x0 = np.zeros(drone.x_dim, dtype=float)
    x0[0] = rand_uniform_sym(rng, x_max)
    x0[1] = rand_uniform_sym(rng, y_max)
    x0[2] = rand_uniform_sym(rng, z_max)
    x0[3] = rand_uniform_sym(rng, phi_max)
    x0[4] = rand_uniform_sym(rng, theta_max)
    x0[5] = 0.0

    return x0