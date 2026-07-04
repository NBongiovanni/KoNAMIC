from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def build_time_grid(dt: float, t_sim: float) -> NDArray[np.float64]:
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")

    if t_sim <= 0:
        raise ValueError(f"t_sim must be > 0, got {t_sim}")

    return np.arange(0.0, t_sim + dt / 2.0, dt)