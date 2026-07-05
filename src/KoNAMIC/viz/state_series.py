from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StatePlotSeries:
    label: str
    values: np.ndarray
    color: str | None = None
    linestyle: str = "-"

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError(
                f"State series {self.label!r} must be 2D, "
                f"got shape {self.values.shape}."
            )
