from __future__ import annotations

import numpy as np

from ..scenario import Scenario
from ..scenario_generator import ScenarioGenerator
from ..signals import build_multistep_signal, low_pass_filter_reference


class CartPoleScenarioGenerator(ScenarioGenerator):
    """
    Scenario generator for CartPole.

    State convention:
        x = [p, theta, p_dot, theta_dot]

    Reference convention:
        reference.shape = (T, 4)
        reference[:, 0] = p_ref
        reference[:, 1] = theta_ref = 0
        reference[:, 2] = p_dot_ref = 0
        reference[:, 3] = theta_dot_ref = 0
    """

    def sample(
        self,
        *,
        time: np.ndarray,
        index: int | None = None,
        num_traj: int | None = None,
    ) -> Scenario:
        profile = self._select_profile(index=index, num_traj=num_traj)

        time = np.asarray(time, dtype=float).reshape(-1)

        x0 = self._sample_initial_condition(profile)
        reference = self._build_controller_reference(
            profile=profile,
            x0=x0,
            time=time,
        )

        return Scenario(
            name=profile,
            x0=x0,
            reference=reference,
            t_final=float(time[-1]),
            metadata={
                "profile": profile,
                "system_name": self.system.system_name,
                "index": index,
            },
        )

    def _sample_initial_condition(self, profile: str) -> np.ndarray:
        initial_conditions = self.cfg.scenario.initial_conditions

        p_max = float(initial_conditions["p_max"])
        theta_max = float(initial_conditions["theta_max"])
        p_dot_max = float(initial_conditions["p_dot_max"])
        theta_dot_max = float(initial_conditions["theta_dot_max"])

        if profile == "balance":
            p_bound = p_max
            theta_bound = theta_max
            p_dot_bound = p_dot_max
            theta_dot_bound = theta_dot_max

        elif profile == "step_p":
            p_bound = p_max
            theta_bound = theta_max
            p_dot_bound = p_dot_max
            theta_dot_bound = theta_dot_max

        else:
            raise ValueError(f"Unsupported CartPole profile: {profile!r}")

        x0 = np.zeros(self.system.x_dim, dtype=float)
        x0[0] = self.rng.uniform(-p_bound, p_bound)
        x0[1] = self.rng.uniform(-theta_bound, theta_bound)
        x0[2] = self.rng.uniform(-p_dot_bound, p_dot_bound)
        x0[3] = self.rng.uniform(-theta_dot_bound, theta_dot_bound)

        return x0

    def _build_controller_reference(
        self,
        *,
        profile: str,
        x0: np.ndarray,
        time: np.ndarray,
    ) -> np.ndarray:
        references = self.cfg.scenario.references

        p_ref_max = float(references["p_max"])
        tau_ref_min = float(references["tau_min"])
        tau_ref_max = float(references["tau_max"])
        n_steps = len(time)

        reference = np.zeros((n_steps, self.system.x_dim), dtype=float)

        if profile == "balance":
            reference[:, 0] = 0.0

        elif profile == "step_p":
            n_segments = int(self.rng.integers(2, 5))
            p_ref = build_multistep_signal(
                n_samples=n_steps,
                n_segments=n_segments,
                max_abs=p_ref_max,
                rng=self.rng,
            )

            p_ref = low_pass_filter_reference(
                ref=p_ref.reshape(-1, 1),
                x0_ref=np.array([x0[0]], dtype=float),
                tau_ref_min=tau_ref_min,
                tau_ref_max=tau_ref_max,
                rng=self.rng,
                dt=self.cfg.dt,
            ).reshape(-1)

            reference[:, 0] = p_ref

        else:
            raise ValueError(f"Unsupported CartPole reference profile: {profile!r}")

        reference[:, 1] = 0.0
        reference[:, 2] = 0.0
        reference[:, 3] = 0.0

        return reference