from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from KoNAMIC.viz.primitives_single import InputPlotGroup
from KoNAMIC.viz.state_series import StatePlotSeries


@dataclass(frozen=True)
class SingleStateInputPlotData:
    time_x: np.ndarray
    state_series: list[StatePlotSeries]
    x_labels: list[str]
    time_u: np.ndarray
    u: np.ndarray
    input_groups: list[InputPlotGroup]

    def __post_init__(self) -> None:
        if not self.state_series:
            raise ValueError("state_series must contain at least one series.")
        if len(self.state_series) > 2:
            raise ValueError(
                "SingleStateInputPlotData currently supports at most two state series, "
                f"got {len(self.state_series)}."
            )

        reference_shape = self.state_series[0].values.shape
        for series in self.state_series[1:]:
            if series.values.shape != reference_shape:
                raise ValueError(
                    "All state series must have the same shape, "
                    f"got {series.values.shape} for {series.label!r} "
                    f"and {reference_shape} for {self.state_series[0].label!r}."
                )

        if len(self.x_labels) != self.n_states:
            raise ValueError(
                f"len(x_labels) must be {self.n_states}, got {len(self.x_labels)}."
            )
        if self.u.ndim != 2:
            raise ValueError(f"u must be 2D, got shape {self.u.shape}.")

    @property
    def n_states(self) -> int:
        return self.state_series[0].values.shape[1]

    @property
    def n_inputs(self) -> int:
        return len(self.input_groups)
