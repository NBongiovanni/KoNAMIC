from __future__ import annotations

import numpy as np

from .config import DataGenerationConfig, Dataset
from .profile import get_profile
from .initial_conditions import sample_initial_condition
from .references import generate_reference_6d
from .simulation import simulate_trajectory
from .post_process import post_process
from .metadata import build_metadata


def generate_dataset(
    *,
    cfg: DataGenerationConfig,
    plant,
    controller_factory,
    split: str
) -> tuple[Dataset, dict]:

    rng = np.random.default_rng(cfg.seed)

    time = np.arange(0.0, cfg.t_sim + cfg.dt / 2.0, cfg.dt)
    n_steps = len(time)

    states = np.zeros((cfg.n_traj, n_steps, 12), dtype=float)
    inputs = np.zeros((cfg.n_traj, n_steps, 4), dtype=float)
    states_ref = np.zeros((cfg.n_traj, n_steps, 6), dtype=float)

    for i in range(cfg.n_traj):
        print(i)
        profile = get_profile(cfg, i)
        x0 = sample_initial_condition(cfg, profile, rng)
        ref6 = generate_reference_6d(cfg, time, profile, x0, rng)

        controller = controller_factory()

        result = simulate_trajectory(
            cfg=cfg,
            plant=plant,
            controller=controller,
            x0=x0,
            ref6=ref6,
            time=time,
        )

        states[i] = result.states
        inputs[i] = result.inputs
        states_ref[i] = result.states_ref

    raw_dataset = Dataset(
        states=states,
        inputs=inputs,
        states_ref=states_ref,
        time=time,
    )

    dataset = post_process(raw_dataset, cfg)
    metadata = build_metadata(dataset, cfg, split)
    return dataset, metadata