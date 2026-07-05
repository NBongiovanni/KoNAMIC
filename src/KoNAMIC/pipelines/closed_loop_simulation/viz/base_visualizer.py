from __future__ import annotations

from typing import Optional

from KoNAMIC.core.systems import SystemSpec
from KoNAMIC.viz.base_visualizer import BaseStateInputVisualizer


class BaseClosedLoopVisualizer(BaseStateInputVisualizer):
    def __init__(
        self,
        system_spec: SystemSpec,
        only_position: bool,
        num_columns_states: int,
        num_columns_inputs: int,
        rc_params: Optional[dict] = None,
    ) -> None:
        super().__init__(
            only_position=only_position,
            num_columns_states=num_columns_states,
            num_columns_inputs=num_columns_inputs,
            rc_params=rc_params,
        )
        self.system_spec = system_spec
