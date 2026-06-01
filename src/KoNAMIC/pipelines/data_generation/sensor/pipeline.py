from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import numpy as np
from tqdm import tqdm

from KoNAMIC.core.drone import DroneSpec
from KoNAMIC.core.plants import Plant
from KoNAMIC.viz import plot_dataset_diagnostics

from .sensor_generation_config import SensorGenerationConfig
from .sensor_split_dataset import SensorSplitDataset
from .trajectory_profiles import get_profile, select_one_traj_per_profile
from .initial_conditions import sample_initial_condition
from .references.api import generate_reference
from .references.controller_reference import build_controller_reference
from .simulation import simulate_trajectory
from .metadata import build_metadata


def generate_all_dataset_splits(
    *,
    cfg: SensorGenerationConfig,
    drone: DroneSpec,
    plant: Plant,
    controller_factory,
    output_path: Path,
    plot_debug: bool = False,
) -> None:

    for split_idx, (split_name, split_info) in enumerate(cfg.splits.items()):
        split_cfg = replace(cfg, seed=cfg.seed + split_idx,)

        dataset, metadata = generate_one_dataset_split(
            config=split_cfg,
            drone=drone,
            plant=plant,
            controller_factory=controller_factory,
            split=split_name,
            num_traj=split_info.num_traj,
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
            metadata=np.array(metadata, dtype=object),
        )

        if plot_debug:
            traj_indices = select_one_traj_per_profile(
                config=split_cfg,
                num_traj=split_info.num_traj,
                drone=drone,
            )

            plot_dataset_diagnostics(
                dataset=dataset,
                drone=drone,
                save_dir=output_path / "diagnostics",
                split_name=split_name,
                traj_indices=traj_indices,
                only_positions=True,
            )


def generate_one_dataset_split(
    config: SensorGenerationConfig,
    drone: DroneSpec,
    plant: Plant,
    controller_factory,
    split: str,
    num_traj: int,
) -> tuple[SensorSplitDataset, dict]:

    rng = np.random.default_rng(config.seed)
    time = np.arange(0.0, config.t_sim + config.dt / 2.0, config.dt)
    n_steps = len(time)

    states = np.zeros((num_traj, n_steps, drone.x_dim), dtype=float)
    inputs = np.zeros((num_traj, n_steps-1, drone.u_dim), dtype=float)
    states_ref = np.zeros((num_traj, n_steps, drone.x_dim), dtype=float)

    for i in tqdm(range(num_traj), desc="Generating sensor trajectories"):
        profile = get_profile(config, i, num_traj, drone)
        x0 = sample_initial_condition(config,profile,rng,drone)

        # Improve these two functions names:
        ref_user = generate_reference(
            cfg=config,
            time=time,
            profile=profile,
            x0=x0,
            rng=rng,
            drone=drone,
        )

        ref_controller = build_controller_reference(ref_user, drone)
        controller = controller_factory()

        result = simulate_trajectory(
            plant=plant,
            controller=controller,
            x0=x0,
            ref_controller=ref_controller,
            time=time,
            drone=drone,
        )

        if result.states.shape != (n_steps, drone.x_dim):
            raise ValueError(
                f"Invalid states shape for trajectory {i}: "
                f"expected {(n_steps, drone.x_dim)}, got {result.states.shape}"
            )

        if result.inputs.shape != (n_steps-1, drone.u_dim):
            raise ValueError(
                f"Invalid inputs shape for trajectory {i}: "
                f"expected {(n_steps, drone.u_dim)}, got {result.inputs.shape}"
            )

        if result.states_ref.shape != (n_steps, drone.x_dim):
            raise ValueError(
                f"Invalid states_ref shape for trajectory {i}: "
                f"expected {(n_steps, drone.x_dim)}, got {result.states_ref.shape}"
            )

        states[i] = result.states
        inputs[i] = result.inputs
        states_ref[i] = result.states_ref

    dataset = SensorSplitDataset(
        states=states,
        inputs=inputs,
        states_ref=states_ref,
        time=time,
    )
    metadata = build_metadata(dataset, config, split, drone)
    return dataset, metadata
