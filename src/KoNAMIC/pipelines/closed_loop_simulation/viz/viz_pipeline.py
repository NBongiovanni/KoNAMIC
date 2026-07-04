#!/usr/bin/env python
from pathlib import Path

from KoNAMIC import utils
from KoNAMIC.core.systems import SystemSpec
from KoNAMIC.core.simulation import ClosedLoopTrajectory
from KoNAMIC.pipelines.closed_loop_simulation.viz.single_visualizer import ClosedLoopSingleVisualizer


def run_closed_loop_visualization(
    simulation_indexes: list[int],
    system_spec: SystemSpec,
    base_control_runs_dir: Path,
    simulation_results: list[ClosedLoopTrajectory],
    only_positions: bool,
    num_columns_states: int,
    num_columns_inputs: int,
) -> None:

    for i in simulation_indexes:
        run_dir = base_control_runs_dir / f"run_{i}"
        run_dir.mkdir(parents=True, exist_ok=True)
        utils.save_sim_result(simulation_results[i], run_dir / "results.pkl")

        ctrl_visualizer = ClosedLoopSingleVisualizer(
            system_spec,
            simulation_results[i],
            run_dir,
            only_positions,
            num_columns_states,
            num_columns_inputs,
        )
        ctrl_visualizer.visualize()