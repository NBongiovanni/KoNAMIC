from dataclasses import asdict

from KoNAMIC.core.systems import SystemSpec
from .sensor_split_dataset import SensorSplitDataset
from .config import SensorGenerationConfig

def build_metadata(
    dataset: SensorSplitDataset,
    cfg: SensorGenerationConfig,
    split: str,
    system_spec: SystemSpec,
) -> dict:

    states_names = system_spec.state_names
    inputs_names = system_spec.input_names
    refs_names = system_spec.ref_names

    return {
        "split": split,
        "drone_dim": system_spec.system_dim,
        "drone_name": system_spec.system_name,

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

        "profile_names": list(dataset.profiles),
        "profile_counts": {
            profile: dataset.profiles.count(profile)
            for profile in sorted(set(dataset.profiles))
        },
    }