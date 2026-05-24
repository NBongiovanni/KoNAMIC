from dataclasses import dataclass, asdict

from .config import DataGenerationConfig, Dataset


def build_metadata(dataset: Dataset, cfg: DataGenerationConfig, split: str) -> dict:
    return {
        "split": split,
        "dt": cfg.dt,
        "t_sim": cfg.t_sim,
        "n_trajs": cfg.n_traj,
        "n_steps": dataset.states.shape[1],
        "downsample_step": cfg.ds_step,
        "smooth_window": cfg.smooth_window,
        "states_names": ["x", "y", "z", "phi", "theta", "psi", "vx", "vy", "vz", "p", "q", "r"],
        "inputs_names": ["T", "tau_x", "tau_y", "tau_z"],
        "refs_names": ["x_ref", "y_ref", "z_ref", "psi_ref", "phi_ref_int", "theta_ref_int"],
        "states_min": dataset.states.min(axis=(0, 1)),
        "states_max": dataset.states.max(axis=(0, 1)),
        "states_mean": dataset.states.mean(axis=(0, 1)),
        "states_std": dataset.states.std(axis=(0, 1)),
        "inputs_min": dataset.inputs.min(axis=(0, 1)),
        "inputs_max": dataset.inputs.max(axis=(0, 1)),
        "refs_min": dataset.states_ref.min(axis=(0, 1)),
        "refs_max": dataset.states_ref.max(axis=(0, 1)),
        "config": asdict(cfg),
    }