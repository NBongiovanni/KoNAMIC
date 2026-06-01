from dataclasses import asdict

from KoNAMIC.core.drone import DroneSpec
from .sensor_generation_config import SensorGenerationConfig
from .sensor_split_dataset import SensorSplitDataset


def build_metadata(
    dataset: SensorSplitDataset, cfg: SensorGenerationConfig, split: str, drone: DroneSpec,
) -> dict:

    states_names = get_state_names(drone)
    inputs_names = get_input_names(drone)
    refs_names = get_ref_names(drone)

    return {
        "split": split,
        "drone_dim": drone.drone_dim,
        "drone_name": drone.name,

        "dt": cfg.dt,
        "t_sim": cfg.t_sim,
        "n_steps": dataset.states.shape[1],

        "states_shape": dataset.states.shape,
        "inputs_shape": dataset.inputs.shape,
        "refs_shape": dataset.states_ref.shape,

        "states_names": states_names,
        "inputs_names": inputs_names,
        "refs_names": refs_names,

        "states_min": dataset.states.min(axis=(0, 1)),
        "states_max": dataset.states.max(axis=(0, 1)),
        "states_mean": dataset.states.mean(axis=(0, 1)),
        "states_std": dataset.states.std(axis=(0, 1)),

        "inputs_min": dataset.inputs.min(axis=(0, 1)),
        "inputs_max": dataset.inputs.max(axis=(0, 1)),
        "inputs_mean": dataset.inputs.mean(axis=(0, 1)),
        "inputs_std": dataset.inputs.std(axis=(0, 1)),

        "refs_min": dataset.states_ref.min(axis=(0, 1)),
        "refs_max": dataset.states_ref.max(axis=(0, 1)),
        "refs_mean": dataset.states_ref.mean(axis=(0, 1)),
        "refs_std": dataset.states_ref.std(axis=(0, 1)),

        "config": asdict(cfg),
    }


def get_state_names(drone) -> list[str]:
    if drone.drone_dim == 2:
        return ["y", "z", "theta", "y_dot", "z_dot", "theta_dot"]

    if drone.drone_dim == 3:
        return ["x", "y", "z", "phi", "theta", "psi", "x_dot", "y_dot", "z_dot", "p", "q", "r"]
    raise ValueError(f"Unsupported drone_dim: {drone.drone_dim}")


def get_input_names(drone) -> list[str]:
    if drone.drone_dim == 2:
        return [
            "F",
            "tau",
        ]

    if drone.drone_dim == 3:
        return [
            "F",
            "tau_x",
            "tau_y",
            "tau_z",
        ]

    raise ValueError(f"Unsupported drone_dim: {drone.drone_dim}")


def get_ref_names(drone) -> list[str]:
    """
    states_ref est stocké avec la même dimension que states.
    Les composantes non directement commandées peuvent correspondre
    à des références internes, par exemple theta_cmd en 2D
    ou phi_cmd/theta_cmd en 3D.
    """
    if drone.drone_dim == 2:
        return [
            "y_ref",
            "z_ref",
            "theta_cmd",
            "y_dot_ref",
            "z_dot_ref",
            "theta_dot_ref",
        ]

    if drone.drone_dim == 3:
        return [
            "x_ref",
            "y_ref",
            "z_ref",
            "phi_cmd",
            "theta_cmd",
            "psi_ref",
            "x_dot_ref",
            "y_dot_ref",
            "z_dot_ref",
            "p_ref",
            "q_ref",
            "r_ref",
        ]

    raise ValueError(f"Unsupported drone_dim: {drone.drone_dim}")