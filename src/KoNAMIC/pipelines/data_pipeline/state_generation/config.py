from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DataGenerationConfig:
    n_traj: int = 10
    dt: float = 1e-3
    t_sim: float = 3.0

    ds_step: int = 10
    smooth_window: int = 10

    x_ref_max: float = 0.5
    y_ref_max: float = 0.5
    z_ref_max: float = 0.5

    x_init_max: float = 0.5
    y_init_max: float = 0.5
    z_init_max: float = 0.5
    angle_init_max: float = np.deg2rad(15.0)

    tau_ref_min: float = 0.05
    tau_ref_max: float = 0.4

    only_aggressive: bool = False
    init_angles_to_zero: bool = False

    seed: int = 0
    config_controller: str = "..."

    split_lengths: dict[str, int] = field(
        default_factory=lambda: {
            "train": 10,
            "val_1": 2,
            "val_2": 2,
        }
    )


@dataclass
class TrajectoryResult:
    states: np.ndarray      # (T, 12)
    inputs: np.ndarray      # (T, 4)
    states_ref: np.ndarray  # (T, 6)
    time: np.ndarray        # (T,)


@dataclass
class Dataset:
    states: np.ndarray      # (N, T, 12)
    inputs: np.ndarray      # (N, T, 4)
    states_ref: np.ndarray  # (N, T, 6)
    time: np.ndarray        # (T,)