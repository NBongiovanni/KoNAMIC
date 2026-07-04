from __future__ import annotations

from typing import List
import numpy as np


def get_angle_indexes() -> List[int]:
    # State convention: [p, theta, p_dot, theta_dot]
    return [1, 3]


def convert_rad_to_deg_np(x, idxs: list[int]) -> np.ndarray:
    x = np.asarray(x).copy()
    x[..., idxs] *= (180.0 / np.pi)
    return x
