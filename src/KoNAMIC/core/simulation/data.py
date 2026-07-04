from dataclasses import dataclass
from typing import Optional, TypeAlias

import numpy as np
import torch

SubLosses: TypeAlias = dict[str, float]


@dataclass
class InputsData:
    """Données de contrôle."""
    u_physical: np.ndarray
    u_scaled: np.ndarray


@dataclass
class TrajectoryData:
    """Données pour une trajectoire avec référence et erreur."""
    traj: np.ndarray | torch.Tensor
    ref_traj: np.ndarray | torch.Tensor
    error: Optional[np.ndarray]

    def __setstate__(self, state):
        self.__dict__.update(state)