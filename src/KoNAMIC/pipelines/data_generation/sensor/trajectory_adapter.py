from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from KoNAMIC.core.simulation import ClosedLoopTrajectory
from KoNAMIC.core.systems import SystemSpec


@dataclass(frozen=True)
class SensorTrajectoryArrays:
    states: np.ndarray
    inputs: np.ndarray
    states_ref: np.ndarray
    time: np.ndarray


def closed_loop_trajectory_to_sensor_arrays(
    result: ClosedLoopTrajectory,
    *,
    system_spec: SystemSpec,
    expected_num_steps: int,
) -> SensorTrajectoryArrays:
    """
    Convertit le format riche ClosedLoopTrajectory vers le format plat attendu
    par la génération de datasets capteurs.

    Convention dataset :
    - states.shape     == (T, x_dim)
    - inputs.shape     == (T - 1, u_dim)
    - states_ref.shape == (T, x_dim)
    - time.shape       == (T,)
    """

    if result.x_data is None:
        raise ValueError("ClosedLoopTrajectory.x_data is None.")

    if result.inputs_data is None:
        raise ValueError("ClosedLoopTrajectory.inputs_data is None.")

    states = np.asarray(result.x_data.traj, dtype=float)
    states_ref = np.asarray(result.x_data.ref_traj, dtype=float)
    inputs = np.asarray(result.inputs_data.u_physical, dtype=float)
    time = np.asarray(result.time, dtype=float)

    expected_state_shape = (expected_num_steps, system_spec.state_dim)
    expected_input_shape = (expected_num_steps - 1, system_spec.input_dim)

    if states.shape != expected_state_shape:
        raise ValueError(
            f"Invalid states shape: expected {expected_state_shape}, "
            f"got {states.shape}."
        )

    if states_ref.shape != expected_state_shape:
        raise ValueError(
            f"Invalid states_ref shape: expected {expected_state_shape}, "
            f"got {states_ref.shape}."
        )

    if inputs.shape != expected_input_shape:
        raise ValueError(
            f"Invalid inputs shape: expected {expected_input_shape}, "
            f"got {inputs.shape}."
        )

    if time.shape != (expected_num_steps,):
        raise ValueError(
            f"Invalid time shape: expected {(expected_num_steps,)}, "
            f"got {time.shape}."
        )

    return SensorTrajectoryArrays(
        states=states,
        inputs=inputs,
        states_ref=states_ref,
        time=time,
    )