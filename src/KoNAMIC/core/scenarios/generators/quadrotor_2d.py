from __future__ import annotations

import numpy as np

from ..scenario import Scenario
from ..scenario_generator import ScenarioGenerator
from ..signals import rand_uniform_sym, build_multistep_signal, low_pass_filter_reference


class Quadrotor2DScenarioGenerator(ScenarioGenerator):
    """
    Scenario generator for the planar quadrotor.

    State convention:
        x = [y, z, theta, y_dot, z_dot, theta_dot]

    Controller reference convention:
        reference.shape = (T, 6)
        reference[:, 0] = y_ref
        reference[:, 1] = z_ref
        reference[:, 2] = theta_ref
        reference[:, 3] = y_dot_ref
        reference[:, 4] = z_dot_ref
        reference[:, 5] = theta_dot_ref
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
        position_max = float(initial_conditions["position_max"])
        angle_max = float(initial_conditions["angle_max"])
        velocity_max = float(initial_conditions["velocity_max"])

        bounds = {
            "hover": (0.0, 0.0, 0.0, 0.0),
            "step_z": (0.0, position_max, 0.0, velocity_max),
            "step_y": (position_max, 0.0, 0.0, velocity_max),
            "step_yz": (position_max, position_max, angle_max, velocity_max),
        }

        if profile not in bounds:
            raise ValueError(f"Unsupported 2D initial-condition profile: {profile}")

        y_bound, z_bound, theta_bound, velocity_bound = bounds[profile]

        x0 = np.zeros(self.system.x_dim, dtype=float)
        x0[0] = rand_uniform_sym(self.rng, y_bound)
        x0[1] = rand_uniform_sym(self.rng, z_bound)
        x0[2] = rand_uniform_sym(self.rng, theta_bound)

        x0[3] = rand_uniform_sym(self.rng, velocity_bound)  # y_dot
        x0[4] = rand_uniform_sym(self.rng, velocity_bound)  # z_dot
        x0[5] = rand_uniform_sym(self.rng, velocity_bound)  # theta_dot

        return x0

    def _build_controller_reference(
        self,
        profile: str,
        x0: np.ndarray,
        time: np.ndarray,
    ) -> np.ndarray:
        compact_ref = self._build_compact_position_reference(
            profile=profile,
            x0=x0,
            n_steps=len(time),
        )

        controller_ref = np.zeros((len(time), self.system.x_dim), dtype=float)
        controller_ref[:, 0] = compact_ref[:, 0]  # y_ref
        controller_ref[:, 1] = compact_ref[:, 1]  # z_ref

        return controller_ref

    def _build_compact_position_reference(
        self,
        profile: str,
        x0: np.ndarray,
        n_steps: int,
    ) -> np.ndarray:
        references = self.cfg.scenario.references

        y_ref_max = float(references["y_max"])
        z_ref_max = float(references["z_max"])

        step_signal = references["step_signal"]
        n_segments_min = int(step_signal["n_segments_min"])
        n_segments_max = int(step_signal["n_segments_max"])

        if n_segments_min <= 0 or n_segments_max <= 0:
            raise ValueError(
                f"n_segments_min and n_segments_max must be positive, got "
                f"n_segments_min={n_segments_min}, n_segments_max={n_segments_max}."
            )

        if n_segments_min > n_segments_max:
            raise ValueError(
                f"n_segments_min must be <= n_segments_max, got "
                f"n_segments_min={n_segments_min}, n_segments_max={n_segments_max}."
            )

        reference_smoothing = references["reference_smoothing"]
        tau_ref_min = float(reference_smoothing["time_constant_min"])
        tau_ref_max = float(reference_smoothing["time_constant_max"])

        ref = np.zeros((n_steps, 2), dtype=float)

        ref[:, 0] = x0[0]
        ref[:, 1] = x0[1]

        # rng.integers high bound is exclusive, hence +1.
        n_segments = int(self.rng.integers(n_segments_min, n_segments_max + 1))

        if profile == "hover":
            pass

        elif profile == "step_z":
            ref[:, 1] = build_multistep_signal(
                n_samples=n_steps,
                n_segments=n_segments,
                max_abs=z_ref_max,
                rng=self.rng,
            )

        elif profile == "step_y":
            ref[:, 0] = build_multistep_signal(
                n_samples=n_steps,
                n_segments=n_segments,
                max_abs=y_ref_max,
                rng=self.rng,
            )

        elif profile == "step_yz":
            ref[:, 0] = build_multistep_signal(
                n_samples=n_steps,
                n_segments=n_segments,
                max_abs=y_ref_max,
                rng=self.rng,
            )
            ref[:, 1] = build_multistep_signal(
                n_samples=n_steps,
                n_segments=n_segments,
                max_abs=z_ref_max,
                rng=self.rng,
            )

        else:
            raise ValueError(f"Unsupported 2D reference profile: {profile}")

        return low_pass_filter_reference(
            ref=ref,
            x0_ref=np.array([x0[0], x0[1]], dtype=float),
            tau_ref_min=tau_ref_min,
            tau_ref_max=tau_ref_max,
            rng=self.rng,
            dt=self.cfg.dt,
        )

    def _select_profile(
        self,
        index: int | None = None,
        num_traj: int | None = None,
    ) -> str:
        profiles = self.cfg.scenario.profiles

        if not profiles:
            raise ValueError("No scenario profiles configured.")

        profile_names = list(profiles.keys())
        weights = np.asarray(list(profiles.values()), dtype=float)

        if np.any(weights < 0.0):
            raise ValueError(f"Profile weights must be non-negative, got {profiles}")

        total_weight = float(np.sum(weights))
        if total_weight <= 0.0:
            raise ValueError(
                f"At least one profile weight must be positive, got {profiles}"
            )

        probabilities = weights / total_weight

        if index is None or num_traj is None:
            return str(self.rng.choice(profile_names, p=probabilities))

        if num_traj <= 0:
            raise ValueError(f"num_traj must be positive, got {num_traj}")

        if index < 0 or index >= num_traj:
            raise ValueError(
                f"index must satisfy 0 <= index < num_traj, "
                f"got index={index}, num_traj={num_traj}"
            )

        cumulative = np.cumsum(probabilities)
        relative_position = (index + 0.5) / num_traj

        profile_idx = int(np.searchsorted(cumulative, relative_position, side="right"))
        profile_idx = min(profile_idx, len(profile_names) - 1)
        return profile_names[profile_idx]