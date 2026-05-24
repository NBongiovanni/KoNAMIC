#!/usr/bin/env python
"""
Main script for executing MPC control simulations.

Execution modes:
- Single simulation with visualization
- Multiple simulations with statistics computation
"""
from pathlib import Path

from KoNAMIC.core.drone import DroneSpec
from KoNAMIC.core.simulation import ClosedLoopTrajectory, save_sim_result
from KoNAMIC.pipelines.closed_loop_simulation.viz.single_visualizer import ClosedLoopSingleVisualizer


def run_closed_loop_visualization(
    simulation_indexes: list[int],
    drone: DroneSpec,
    control_params: dict,
    base_ctrl_runs_dir: Path,
    simulation_results: list[ClosedLoopTrajectory],
    only_positions: bool,
    num_columns: int,
) -> None:
    for i in simulation_indexes:
        run_dir = base_ctrl_runs_dir / f"run_{i}"
        run_dir.mkdir(parents=True, exist_ok=True)
        save_sim_result(simulation_results[i], run_dir / "results.pkl")

        ctrl_visualizer = ClosedLoopSingleVisualizer(
            drone.drone_dim,
            simulation_results[i],
            run_dir,
            control_params["dt"],
            control_params["use_nominal_plant"],
            only_positions,
            num_columns,
            num_columns,
        )
        ctrl_visualizer.visualize()