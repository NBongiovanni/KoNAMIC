from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import numpy as np

from KoNAMIC.core.drone import DroneSpec
from KoNAMIC.core.plants import Plant
from KoNAMIC.viz import plot_dataset_diagnostics

from .sensor_generation_config import SensorGenerationConfig
from .dataset import Dataset
from .profile import get_profile
from .initial_conditions import sample_initial_condition
from .references import generate_reference
from .simulation import simulate_trajectory
from .metadata import build_metadata


def generate_dataset(
    config: SensorGenerationConfig,
    drone: DroneSpec,
    plant: Plant,
    controller_factory,
    split: str,
    num_traj: int,
) -> tuple[Dataset, dict]:

    rng = np.random.default_rng(config.seed)

    time = np.arange(0.0, config.t_sim + config.dt / 2.0, config.dt)
    n_steps = len(time)

    states = np.zeros((num_traj, n_steps, drone.x_dim), dtype=float)
    inputs = np.zeros((num_traj, n_steps, drone.u_dim), dtype=float)
    states_ref = np.zeros((num_traj, n_steps, drone.x_dim), dtype=float)

    for i in range(num_traj):
        print(i)
        profile = get_profile(
            cfg=config,
            traj_idx=i,
            num_traj=num_traj,
            drone=drone,
        )
        x0 = sample_initial_condition(
            cfg=config,
            profile=profile,
            rng=rng,
            drone=drone,
        )

        ref_user = generate_reference(
            cfg=config,
            time=time,
            profile=profile,
            x0=x0,
            rng=rng,
            drone=drone,
        )

        ref_controller = build_controller_reference(
            ref_user=ref_user,
            drone=drone,
        )

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

        if result.inputs.shape != (n_steps, drone.u_dim):
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

    dataset = Dataset(
        states=states,
        inputs=inputs,
        states_ref=states_ref,
        time=time,
    )
    metadata = build_metadata(dataset, config, split, drone)
    return dataset, metadata


def build_controller_reference(
    *,
    ref_user: np.ndarray,
    drone,
) -> np.ndarray:
    ref_user = np.asarray(ref_user, dtype=float)

    if ref_user.ndim != 2:
        raise ValueError(
            f"ref_user must be 2D, got shape {ref_user.shape}"
        )

    n_steps = ref_user.shape[0]
    ref_controller = np.zeros((n_steps, drone.x_dim), dtype=float)

    if drone.drone_dim == 2:
        # Convention actuelle d'après vos labels :
        # état 2D = [y, z, theta, y_dot, z_dot, theta_dot]
        #
        # ref_user = [y_ref, z_ref]
        ref_controller[:, 0] = ref_user[:, 0]
        ref_controller[:, 1] = ref_user[:, 1]
        return ref_controller

    if drone.drone_dim == 3:
        # état 3D = [x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, p, q, r]
        #
        # ref_user = [x_ref, y_ref, z_ref]
        ref_controller[:, 0] = ref_user[:, 0]
        ref_controller[:, 1] = ref_user[:, 1]
        ref_controller[:, 2] = ref_user[:, 2]
        return ref_controller

    raise ValueError(f"Unsupported drone_dim: {drone.drone_dim}")


def generate_all_splits(
    *,
    cfg: SensorGenerationConfig,
    drone: DroneSpec,
    plant,
    controller_factory,
    output_path: Path,
    plot_debug: bool = False,
) -> None:

    for split_idx, (split_name, split_info) in enumerate(cfg.splits.items()):
        split_cfg = replace(
            cfg,
            seed=cfg.seed + split_idx,
        )

        dataset, metadata = generate_dataset(
            config=split_cfg,
            drone=drone,
            plant=plant,
            controller_factory=controller_factory,
            split=split_name,
            num_traj=split_info.num_traj,
        )

        save_dataset_npz(
            dataset,
            metadata,
            output_path / f"{split_name}.npz",
        )

        if plot_debug:
            plot_dataset_diagnostics(
                dataset=dataset,
                drone=drone,
                save_dir=output_path / "diagnostics",
                split_name=split_name,
                traj_indices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                only_positions=True,
            )


def save_dataset_npz(dataset: Dataset, metadata: dict, final_path: Path) -> None:
    # TODO: move this function
    final_path.parent.mkdir(parents=True, exist_ok=True)
    print(final_path)

    np.savez_compressed(
        final_path,
        states=dataset.states,
        inputs=dataset.inputs,
        statesRef=dataset.states_ref,
        timeVec=dataset.time,
        metadata=np.array(metadata, dtype=object),
    )