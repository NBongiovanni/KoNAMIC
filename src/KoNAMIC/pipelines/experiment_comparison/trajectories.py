from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectoryComparisonResult:
    time: np.ndarray
    reference_state: np.ndarray
    compared_state: np.ndarray
    input_time: np.ndarray
    inputs: np.ndarray
    dt: float
