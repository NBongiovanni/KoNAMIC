from __future__ import annotations

from typing import Optional

from KoNAMIC.viz.base_visualizer import BaseStateInputVisualizer


class BaseOpenLoopVisualizer(BaseStateInputVisualizer):
    def __init__(
        self,
        system_dim: int,
        dt: float,
        only_position: bool,
        num_columns_states: int,
        num_columns_inputs: int,
        rc_params: Optional[dict] = None,
    ) -> None:
        assert system_dim in (2, 3)

        super().__init__(
            only_position=only_position,
            num_columns_states=num_columns_states,
            num_columns_inputs=num_columns_inputs,
            rc_params=rc_params,
        )
        self.drone_dim = int(system_dim)
        self.dt = float(dt)
