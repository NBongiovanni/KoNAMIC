from __future__ import annotations

import numpy as np

from ..scenario import Scenario
from ..scenario_generator import ScenarioGenerator
from ..signals import rand_uniform_sym, build_multistep_signal, low_pass_filter_reference


class Quadrotor3DScenarioGenerator(ScenarioGenerator):
    """
    Scenario generator for the 3D quadrotor.

    State convention:
        x = [
            x, y, z,
            phi, theta, psi,
            x_dot, y_dot, z_dot,
            phi_dot, theta_dot, psi_dot,
        ]

    Controller reference convention:
        reference.shape = (T, 12)
        reference[:, 0] = x_ref
        reference[:, 1] = y_ref
        reference[:, 2] = z_ref

    The remaining components are set to zero:
        roll/pitch/yaw references,
        linear velocity references,
        angular velocity references.
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

        position_bounds = {
            "hover": (0.0, 0.0, 0.0),
            "step_x": (position_max, 0.0, 0.0),
            "step_y": (0.0, position_max, 0.0),
            "step_z": (0.0, 0.0, position_max),
            "step_xy": (position_max, position_max, 0.0),
            "step_xz": (position_max, 0.0, position_max),
            "step_yz": (0.0, position_max, position_max),
            "step_xyz": (position_max, position_max, position_max),
        }

        if profile not in position_bounds:
            raise ValueError(f"Unsupported 3D initial-condition profile: {profile}")

        x_bound, y_bound, z_bound = position_bounds[profile]

        x0 = np.zeros(self.system.x_dim, dtype=float)

        # Position
        x0[0] = rand_uniform_sym(self.rng, x_bound)
        x0[1] = rand_uniform_sym(self.rng, y_bound)
        x0[2] = rand_uniform_sym(self.rng, z_bound)

        # Attitude
        if profile == "hover":
            attitude_bound = 0.0
        else:
            attitude_bound = angle_max

        x0[3] = rand_uniform_sym(self.rng, attitude_bound)  # phi
        x0[4] = rand_uniform_sym(self.rng, attitude_bound)  # theta
        x0[5] = rand_uniform_sym(self.rng, attitude_bound)  # psi

        # Linear velocities
        x0[6] = rand_uniform_sym(self.rng, velocity_max)  # x_dot
        x0[7] = rand_uniform_sym(self.rng, velocity_max)  # y_dot
        x0[8] = rand_uniform_sym(self.rng, velocity_max)  # z_dot

        # Angular velocities
        x0[9] = rand_uniform_sym(self.rng, velocity_max)   # phi_dot
        x0[10] = rand_uniform_sym(self.rng, velocity_max)  # theta_dot
        x0[11] = rand_uniform_sym(self.rng, velocity_max)  # psi_dot

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
        controller_ref[:, 0] = compact_ref[:, 0]  # x_ref
        controller_ref[:, 1] = compact_ref[:, 1]  # y_ref
        controller_ref[:, 2] = compact_ref[:, 2]  # z_ref

        return controller_ref

    def _build_compact_position_reference(
        self,
        profile: str,
        x0: np.ndarray,
        n_steps: int,
    ) -> np.ndarray:
        references = self.cfg.scenario.references

        x_ref_max = float(references["x_max"])
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

        ref = np.zeros((n_steps, 3), dtype=float)
        ref[:, 0] = x0[0]
        ref[:, 1] = x0[1]
        ref[:, 2] = x0[2]

        # rng.integers high bound is exclusive, hence +1.
        n_segments = int(self.rng.integers(n_segments_min, n_segments_max + 1))

        if profile == "hover":
            pass

        elif profile == "step_x":
            ref[:, 0] = self._build_axis_reference(n_steps, n_segments, x_ref_max)

        elif profile == "step_y":
            ref[:, 1] = self._build_axis_reference(n_steps, n_segments, y_ref_max)

        elif profile == "step_z":
            ref[:, 2] = self._build_axis_reference(n_steps, n_segments, z_ref_max)

        elif profile == "step_xy":
            ref[:, 0] = self._build_axis_reference(n_steps, n_segments, x_ref_max)
            ref[:, 1] = self._build_axis_reference(n_steps, n_segments, y_ref_max)

        elif profile == "step_xz":
            ref[:, 0] = self._build_axis_reference(n_steps, n_segments, x_ref_max)
            ref[:, 2] = self._build_axis_reference(n_steps, n_segments, z_ref_max)

        elif profile == "step_yz":
            ref[:, 1] = self._build_axis_reference(n_steps, n_segments, y_ref_max)
            ref[:, 2] = self._build_axis_reference(n_steps, n_segments, z_ref_max)

        elif profile == "step_xyz":
            ref[:, 0] = self._build_axis_reference(n_steps, n_segments, x_ref_max)
            ref[:, 1] = self._build_axis_reference(n_steps, n_segments, y_ref_max)
            ref[:, 2] = self._build_axis_reference(n_steps, n_segments, z_ref_max)

        else:
            raise ValueError(f"Unsupported 3D reference profile: {profile}")

        return low_pass_filter_reference(
            ref=ref,
            x0_ref=np.array([x0[0], x0[1], x0[2]], dtype=float),
            tau_ref_min=tau_ref_min,
            tau_ref_max=tau_ref_max,
            rng=self.rng,
            dt=self.cfg.dt,
        )

    def _build_axis_reference(
        self,
        n_steps: int,
        n_segments: int,
        max_abs: float,
    ) -> np.ndarray:
        return build_multistep_signal(
            n_samples=n_steps,
            n_segments=n_segments,
            max_abs=max_abs,
            rng=self.rng,
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