from typing import Optional
from dataclasses import dataclass

import numpy as np

from .data import TrajectoryData, InputsData


@dataclass
class ClosedLoopTrajectory:
    time: np.ndarray
    x_init: np.ndarray
    x_data: Optional[TrajectoryData]
    z_data: TrajectoryData
    im_data: TrajectoryData
    inputs_data: InputsData

    def __setstate__(self, state):
        # Compatibilité anciens pickles
        if "time" not in state and "time_vec" in state:
            state["time"] = state.pop("time_vec")

        if "inputs_data" not in state and "control_data" in state:
            state["inputs_data"] = state.pop("control_data")

        if "x_init" not in state:
            state["x_init"] = None

        self.__dict__.update(state)

