#!/usr/bin/env python
from __future__ import annotations

from typing import Any
import numpy as np

from KoNAMIC.core.drone import DroneSpec
from KoNAMIC.core.simulation import ClosedLoopTrajectory

from .stats.process_stats import process_statistics
from .stats.multirun_metrics import MultiRunMetrics
from .scenarios.initial_conditions import create_state_init_conditions


def run_closed_loop_simulations(
    num_simulations: int,
    control_params: dict,
    controller: Any,
    simulator: Any,
    reference_builder_factory: Any,
    drone: DroneSpec,
    u_eq = None,
) -> list[ClosedLoopTrajectory]:

    metrics = MultiRunMetrics()
    simulation_results: list[ClosedLoopTrajectory] = []
    controller.build()

    for i in range(num_simulations):
        print(f"\n{'=' * 60}")
        print(f"Simulation {i + 1}/{num_simulations}")
        print(f"{'=' * 60}")
        x_init = create_state_init_conditions(drone.x_dim, control_params)

        ref_builder = reference_builder_factory()
        reference_data = ref_builder.build()
        controller.reset()

        if u_eq is not None and hasattr(controller, "set_operating_point"):
            controller.set_operating_point(np.asarray(u_eq, dtype=float))

        _set_controller_reference(controller, reference_data)

        simulation_output = simulator.run(x_init)
        simulation_results.append(simulation_output)

        sqp_iters = _get_controller_sqp_iters(controller)
        stats = process_statistics(simulation_output, sqp_iters)
        print(stats.short_summary())
        metrics.add(stats)

    metrics.display(num_simulations)
    return simulation_results


def _set_controller_reference(controller: Any, reference_data: Any) -> None:
    if isinstance(reference_data, tuple) and len(reference_data) == 3:
        state_ref_traj, im_ref_traj, z_ref_traj = reference_data
        controller.set_reference(z_ref_traj)
        controller.x_ref_traj = state_ref_traj
        return

    controller.set_reference(reference_data)


def _get_controller_sqp_iters(controller: Any) -> list[int]:
    backend = getattr(controller, "backend", None)
    if backend is None:
        return []

    sqp_iters = getattr(backend, "sqp_iters", None)
    if sqp_iters is None:
        return []

    return sqp_iters