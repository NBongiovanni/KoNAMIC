from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

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