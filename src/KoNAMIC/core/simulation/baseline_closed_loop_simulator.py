from __future__ import annotations

import numpy as np

from KoNAMIC.core.control.controllers import BaseController
from KoNAMIC.core.systems import SystemSpec
from KoNAMIC.core.plants import Plant
from KoNAMIC.core.scenarios import Scenario
from .trajectories import ClosedLoopTrajectory
from .time_grid import build_time_grid
from .closed_loop_simulator import ClosedLoopSimulator
from .postprocessing import build_closed_loop_trajectory


class BaselineClosedLoopSimulator(ClosedLoopSimulator):
    def __init__(
        self,
        *,
        system_spec: SystemSpec,
        plant: Plant,
        controller: BaseController,
        dt: float,
        t_sim: float,
    ) -> None:
        self.system_spec = system_spec
        self.plant = plant
        self.controller = controller
        self.dt = dt
        self.t_sim = t_sim
        self.time = build_time_grid(self.dt, self.t_sim)
        self.num_steps = len(self.time)

    def run(self, scenario: Scenario):
        return simulate_trajectory(
            system_spec=self.system_spec,
            plant=self.plant,
            controller=self.controller,
            scenario=scenario,
            time=self.time,
        )


def simulate_trajectory(
    *,
    system_spec: SystemSpec,
    plant: Plant,
    controller: BaseController,
    scenario: Scenario,
    time: np.ndarray,
)-> ClosedLoopTrajectory:
    """
    Simulate one closed-loop trajectory from a Scenario.


    The Scenario contains:
        - x0: initial condition
        - reference: controller-ready reference trajectory
        - metadata: optional information about profile, target, etc.

    scenario.reference must already be in the format expected by the controller,
    with shape (T, system_spec.x_dim).
    """
    n_steps = len(time)

    x0 = np.asarray(scenario.x0, dtype=float).reshape(-1)
    ref_controller = np.asarray(scenario.reference, dtype=float)

    if x0.shape != (system_spec.x_dim,):
        raise ValueError(
            f"x0 must have shape ({system_spec.x_dim},), got {x0.shape} "
            f"for scenario {scenario.name!r}"
        )

    if ref_controller.shape != (n_steps, system_spec.x_dim):
        raise ValueError(
            f"scenario.reference must have shape {(n_steps, system_spec.x_dim)}, "
            f"got {ref_controller.shape} for scenario {scenario.name!r}"
        )

    states = np.zeros((n_steps, system_spec.x_dim), dtype=float)
    inputs = np.zeros((n_steps - 1, system_spec.u_dim), dtype=float)
    states_ref = ref_controller.copy()

    states[0] = x0

    controller.reset()
    controller.set_reference(ref_controller)
    controller.set_initial_conditions(x0)

    x_k = x0.copy()

    for k in range(1, n_steps):
        u_k = controller.compute_control({"x_k": x_k})
        u_k = np.asarray(u_k, dtype=float).reshape(-1)

        if u_k.shape != (system_spec.u_dim,):
            raise ValueError(
                f"Controller returned control with shape {u_k.shape}, "
                f"expected ({system_spec.u_dim},) "
                f"at step {k} for scenario {scenario.name!r}"
            )

        x_next = plant.update_state(x_k, u_k)
        x_next = np.asarray(x_next, dtype=float).reshape(-1)

        if x_next.shape != (system_spec.x_dim,):
            raise ValueError(
                f"Plant returned state with shape {x_next.shape}, "
                f"expected ({system_spec.x_dim},) "
                f"at step {k} for scenario {scenario.name!r}"
            )

        states[k] = x_next
        inputs[k - 1] = u_k

        _update_effective_state_reference(
            states_ref=states_ref,
            controller=controller,
            k=k,
            x_dim=system_spec.x_dim,
            scenario_name=scenario.name,
        )

        x_k = x_next

    return build_closed_loop_trajectory(
        time=time,
        x_init=x0,
        x_traj=states,
        x_ref_traj=states_ref,
        u_physical=inputs,
    )


def _update_effective_state_reference(
    *,
    states_ref: np.ndarray,
    controller: BaseController,
    k: int,
    x_dim: int,
    scenario_name: str,
) -> None:
    """
    If available, replace states_ref[k] with the effective state reference
    internally used by the controller.

    This is useful for controllers that augment or modify the reference,
    for example quadrotor controllers computing attitude references from
    position references.
    """
    x_ref_traj = getattr(controller, "x_ref_traj", None)

    if x_ref_traj is None:
        return

    x_ref_traj = np.asarray(x_ref_traj, dtype=float)

    if x_ref_traj.ndim != 2:
        raise ValueError(
            f"controller.x_ref_traj must be 2D, got shape {x_ref_traj.shape} "
            f"for scenario {scenario_name!r}"
        )

    if x_ref_traj.shape[1] != x_dim:
        raise ValueError(
            f"controller.x_ref_traj must have second dimension {x_dim}, "
            f"got {x_ref_traj.shape[1]} for scenario {scenario_name!r}"
        )

    kk = min(k - 1, x_ref_traj.shape[0] - 1)
    states_ref[k] = x_ref_traj[kk]