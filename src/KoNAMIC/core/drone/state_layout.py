from __future__ import annotations

from typing import List
import numpy as np


def get_angle_indexes(drone_dim: int) -> List:
    if drone_dim == 2:
       angle_indexes = [2, 5]
    elif drone_dim == 3:
        angle_indexes = [3, 4, 5, 9, 10, 11]
    else:
        raise ValueError(f"Drone dimension {drone_dim} not supported.")
    return angle_indexes


def convert_rad_to_deg_np(x, idxs: list[int]) -> np.ndarray:
    x = np.asarray(x).copy()
    x[..., idxs] *= (180.0 / np.pi)
    return x