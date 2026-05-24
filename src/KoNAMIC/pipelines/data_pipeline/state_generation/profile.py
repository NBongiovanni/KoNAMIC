from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import numpy as np

from .config import Dataset, DataGenerationConfig

Profile = Literal["hover", "step_z", "step_x", "step_y", "step_xyz"]

def get_profile(cfg: DataGenerationConfig, traj_idx: int) -> Profile:
    if cfg.only_aggressive:
        return "step_xyz"

    n = cfg.n_traj

    if traj_idx < n / 10:
        return "hover"
    if traj_idx < 2 * n / 10:
        return "step_z"
    if traj_idx < 3 * n / 10:
        return "step_x"
    if traj_idx < 4 * n / 10:
        return "step_y"

    return "step_xyz"
