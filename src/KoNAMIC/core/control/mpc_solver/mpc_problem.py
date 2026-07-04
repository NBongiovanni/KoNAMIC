from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Any

import numpy as np


@dataclass
class MPCProblem:
    dt: float
    N: int
    state_dim: int
    u_dim: int

    Q: np.ndarray
    Qf: np.ndarray
    R: np.ndarray
    S: Optional[np.ndarray] = None

    use_input_constraints: bool = False
    u_min: Optional[np.ndarray] = None
    u_max: Optional[np.ndarray] = None

    f_discrete: Optional[Any] = None
    reference_provider: Optional[Callable[[float, Any], Any]] = None
    u_guess: Optional[np.ndarray] = None