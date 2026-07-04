from __future__ import annotations
from dataclasses import dataclass

import numpy as np

@dataclass
class TrajectoryResult:
    states: np.ndarray      # (T, 12)
    inputs: np.ndarray      # (T, 4)
    states_ref: np.ndarray  # (T, 6)
    time: np.ndarray        # (T,)


@dataclass
class SensorSplitDataset:
    states: np.ndarray
    inputs: np.ndarray
    states_ref: np.ndarray
    time: np.ndarray
    profiles: tuple[str, ...]