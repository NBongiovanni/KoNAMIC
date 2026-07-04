from __future__ import annotations

from typing import Any, TextIO
import numpy as np
from tqdm import tqdm

from KoNAMIC.core.simulation import ClosedLoopTrajectory
from KoNAMIC.core.scenarios import ScenarioGenerator

from .stats.process_stats import process_statistics
from .stats.multirun_metrics import MultiRunMetrics

from KoNAMIC.core.simulation import ClosedLoopSimulator


def run_closed_loop_simulations(
    num_simulations: int,
    controller: Any,
    simulator: ClosedLoopSimulator,
    scenario_generator: ScenarioGenerator,
    u_eq=None,
    progress_file: TextIO | None = None,
) -> list[ClosedLoopTrajectory]:
    """
    This routine is the closed-loop evaluation loop: it samples one scenario
    per rollout, resets the controller, runs the simulator, and records the
    resulting trajectory. Scenario generation, controller state, and plant
    simulation are kept separate so each piece can be tested independently.
    The returned trajectories are the source of truth for downstream
    visualization and aggregate metrics.
    """

    time = simulator.time

    metrics = MultiRunMetrics()
    simulation_results: list[ClosedLoopTrajectory] = []

    controller.build()

    simulation_indexes = range(num_simulations)
    if progress_file is not None:
        simulation_indexes = tqdm(
            simulation_indexes,
            total=num_simulations,
            desc="closed-loop eval",
            file=progress_file,
            leave=False,
        )

    for i in simulation_indexes:
        print(f"\n{'=' * 60}")
        print(f"Simulation {i + 1}/{num_simulations}")
        print(f"{'=' * 60}")

        # ------------------------------------------------------------
        # Scenario (FULL OBJECT, not decomposed)
        # ------------------------------------------------------------
        scenario = scenario_generator.sample(
            index=i,
            num_traj=num_simulations,
            time=time,
        )

        # ------------------------------------------------------------
        # Controller reset
        # ------------------------------------------------------------
        controller.reset()

        if u_eq is not None and hasattr(controller, "set_operating_point"):
            controller.set_operating_point(np.asarray(u_eq, dtype=float))

        # ------------------------------------------------------------
        # Simulation (clean separation)
        # ------------------------------------------------------------
        simulation_output = simulator.run(scenario=scenario)

        simulation_results.append(simulation_output)

        # ------------------------------------------------------------
        # Metrics
        # ------------------------------------------------------------
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