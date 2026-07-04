from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from tqdm import tqdm

from KoNAMIC.config.config_utils import save_yaml

from KoNAMIC.core.control.controllers import BaseController
from KoNAMIC.core.systems import SystemSpec
from KoNAMIC.core.plants import Plant
from KoNAMIC.core.scenarios import ScenarioGenerator
from KoNAMIC.core.simulation import BaselineClosedLoopSimulator, build_time_grid
from KoNAMIC.viz import plot_dataset_diagnostics
from .sensor_split_dataset import SensorSplitDataset
from .diagnostic_selections import select_one_traj_per_profile
from .metadata import build_metadata
from .config import SensorGenerationConfig
from .trajectory_adapter import closed_loop_trajectory_to_sensor_arrays


def generate_dataset_splits(
    *,
    sensor_gen_config: SensorGenerationConfig,
    scenario_generator: ScenarioGenerator,
    system_spec: SystemSpec,
    plant: Plant,
    controller: BaseController,
    output_path: Path,
    seed: int,
    plot_debug: bool = False,
) -> None:

    save_yaml(
        sensor_gen_config.to_dict(),
        "generation_config.yaml",
        output_path,
    )

    time = build_time_grid(sensor_gen_config.dt, sensor_gen_config.t_sim)

    simulator = BaselineClosedLoopSimulator(
        system_spec=system_spec,
        plant=plant,
        controller=controller,
        dt=sensor_gen_config.dt,
        t_sim=sensor_gen_config.t_sim,
    )

    for split_idx, (split_name, split_info) in enumerate(sensor_gen_config.splits.items()):
        split_cfg = replace(sensor_gen_config, seed=seed + split_idx)

        dataset, metadata = generate_dataset_split(
            config=split_cfg,
            simulator=simulator,
            system_spec=system_spec,
            scenario_generator=scenario_generator,
            split=split_name,
            num_traj=split_info.num_traj,
            time=time,
        )

        file_path = output_path / f"{split_name}.npz"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        print(file_path)

        np.savez_compressed(
            file_path,
            states=dataset.states,
            inputs=dataset.inputs,
            statesRef=dataset.states_ref,
            timeVec=dataset.time,
            profiles=np.array(dataset.profiles, dtype=object),
            metadata=np.array(metadata, dtype=object),
        )

        if plot_debug:
            traj_indices = select_one_traj_per_profile(dataset.profiles)

            plot_dataset_diagnostics(
                dataset=dataset,
                system=system_spec,
                save_dir=output_path / "diagnostics",
                split_name=split_name,
                traj_indices=traj_indices,
                only_positions=False,
                num_columns_states=2,
                num_columns_inputs=1,
            )


def generate_dataset_split(
    config: SensorGenerationConfig,
    scenario_generator: ScenarioGenerator,
    simulator: BaselineClosedLoopSimulator,
    system_spec: SystemSpec,
    split: str,
    num_traj: int,
    time: np.ndarray,
) -> tuple[SensorSplitDataset, dict]:

    n_steps = len(time)
    states = np.zeros((num_traj, n_steps, system_spec.state_dim), dtype=float)
    inputs = np.zeros((num_traj, n_steps - 1, system_spec.input_dim), dtype=float)
    states_ref = np.zeros((num_traj, n_steps, system_spec.state_dim), dtype=float)
    profiles: list[str] = []

    for i in tqdm(range(num_traj), desc=f"Generating {split} sensor trajectories"):
        scenario = scenario_generator.sample(
            index=i,
            num_traj=num_traj,
            time=time,
        )

        profile = scenario.metadata.get("profile")
        if profile is None:
            raise KeyError(
                f"Scenario {i} has no 'profile' entry in metadata. "
                f"Scenario name: {scenario.name!r}"
            )
        profiles.append(str(profile))
        closed_loop_result = simulator.run(scenario)

        traj = closed_loop_trajectory_to_sensor_arrays(
            closed_loop_result,
            system_spec=system_spec,
            expected_num_steps=n_steps,
        )

        states[i] = traj.states
        inputs[i] = traj.inputs
        states_ref[i] = traj.states_ref

    dataset = SensorSplitDataset(
        states=states,
        inputs=inputs,
        states_ref=states_ref,
        time=time,
        profiles=tuple(profiles),
    )

    metadata = build_metadata(dataset, config, split, system_spec)
    return dataset, metadata