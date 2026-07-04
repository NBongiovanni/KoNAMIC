from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Scenario:
    """
    Simulation scenario shared by dataset generation and closed-loop evaluation.

    Contract:
        x0: shape (x_dim,)
        reference:
            - baseline controllers: shape (T, x_dim)
            - Koopman controllers: may be richer, e.g. tuple(state_ref, im_ref, z_ref)
        t_final: final simulation time
        metadata: optional information
    """

    name: str
    x0: np.ndarray
    reference: Any
    t_final: float
    metadata: dict[str, Any] = field(default_factory=dict)