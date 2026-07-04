from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from KoNAMIC.core.control.config.kmpc_config import KmpcCostConfig
from KoNAMIC.core.systems.system_spec import SystemSpec


KmpcCostMode = Literal["state_in_z", "position_in_z", "full_latent", "structured_latent"]


@dataclass(frozen=True, kw_only=True)
class KmpcWeightMatrices:
    Q: np.ndarray
    Qf: np.ndarray
    R: np.ndarray
    S: np.ndarray


@dataclass(frozen=True, kw_only=True)
class KmpcWeightBuilder:
    cost: KmpcCostConfig
    system_spec: SystemSpec
    z_dim: int
    u_dim: int
    x_in_z: bool

    def build(self) -> KmpcWeightMatrices:
        mode = self._require_mode()

        if mode == "state_in_z":
            Q, Qf = self._build_state_in_z_weights()
        elif mode == "position_in_z":
            Q, Qf = self._build_position_in_z_weights()
        elif mode == "full_latent":
            Q, Qf = self._build_full_latent_weights()
        elif mode == "structured_latent":
            Q, Qf = self._build_structured_latent_weights()
        else:
            raise ValueError(f"Unsupported KMPC cost mode: {mode!r}.")

        R = self._diag_from_vector(self.cost.R, self.u_dim, "controller.cost.R")
        S = self._diag_from_vector(self.cost.S, self.u_dim, "controller.cost.S")

        return KmpcWeightMatrices(Q=Q, Qf=Qf, R=R, S=S)

    def _build_state_in_z_weights(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.x_in_z:
            raise ValueError("controller.cost.mode='state_in_z' requires include_state_in_z=True.")

        x_dim = self.system_spec.state_dim
        if x_dim > self.z_dim:
            raise ValueError(f"system state_dim={x_dim} cannot exceed z_dim={self.z_dim}.")

        q_state = self._require_scalar("Q_state")
        p_state = self._require_scalar("P_state")

        Q = np.zeros((self.z_dim, self.z_dim), dtype=float)
        Qf = np.zeros((self.z_dim, self.z_dim), dtype=float)
        Q[:x_dim, :x_dim] = q_state * np.eye(x_dim)
        Qf[:x_dim, :x_dim] = p_state * np.eye(x_dim)

        return Q, Qf

    def _build_position_in_z_weights(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.x_in_z:
            raise ValueError("controller.cost.mode='position_in_z' requires include_state_in_z=True.")

        x_dim = self.system_spec.state_dim
        if x_dim != 6:
            raise ValueError(
                "controller.cost.mode='position_in_z' currently assumes the 2D "
                f"state layout (y, z, theta, y_dot, z_dot, theta_dot), got state_dim={x_dim}."
            )
        if x_dim > self.z_dim:
            raise ValueError(f"system state_dim={x_dim} cannot exceed z_dim={self.z_dim}.")

        q_pos = self._require_scalar("Q_positions")
        p_pos = self._require_scalar("P_positions")
        q_other_state = self._require_scalar("Q_other_state")
        p_other_state = self._require_scalar("P_other_state")
        q_latent = self._require_scalar("Q_latent")
        p_latent = self._require_scalar("P_latent")

        Q = q_latent * np.eye(self.z_dim, dtype=float)
        Qf = p_latent * np.eye(self.z_dim, dtype=float)

        Q[:x_dim, :x_dim] = q_other_state * np.eye(x_dim)
        Qf[:x_dim, :x_dim] = p_other_state * np.eye(x_dim)

        # State layout: (y, z, theta, y_dot, z_dot, theta_dot).
        # Track only the linear positions strongly; keep attitude, velocities,
        # and learned latent coordinates as soft regularization terms.
        linear_position_indices = (0, 1)
        for idx in linear_position_indices:
            Q[idx, idx] = q_pos
            Qf[idx, idx] = p_pos

        return Q, Qf

    def _build_full_latent_weights(self) -> tuple[np.ndarray, np.ndarray]:
        qz = self._require_scalar("Qz")
        pz = self._require_scalar("Pz")

        return qz * np.eye(self.z_dim), pz * np.eye(self.z_dim)

    def _build_structured_latent_weights(self) -> tuple[np.ndarray, np.ndarray]:
        q_pos = self._require_scalar("Q_positions")
        q_vel = self._require_scalar("Q_velocities")
        p_pos = self._require_scalar("P_positions")
        p_vel = self._require_scalar("P_velocities")

        if self.z_dim % 2 != 0:
            raise ValueError(
                "controller.cost.mode='structured_latent' requires an even z_dim "
                f"to split positions and velocities, got z_dim={self.z_dim}."
            )

        half = self.z_dim // 2
        Q = np.zeros((self.z_dim, self.z_dim), dtype=float)
        Qf = np.zeros((self.z_dim, self.z_dim), dtype=float)

        Q[:half, :half] = q_pos * np.eye(half)
        Q[half:, half:] = q_vel * np.eye(half)
        Qf[:half, :half] = p_pos * np.eye(half)
        Qf[half:, half:] = p_vel * np.eye(half)

        return Q, Qf

    def _require_mode(self) -> KmpcCostMode:
        mode = self.cost.mode
        if mode not in ("state_in_z", "position_in_z", "full_latent", "structured_latent"):
            raise ValueError(
                "controller.cost.mode must be one of: "
                "'state_in_z', 'position_in_z', 'full_latent', 'structured_latent'."
            )
        return mode

    def _require_scalar(self, name: str) -> float:
        value = getattr(self.cost, name)
        if value is None:
            raise ValueError(f"controller.cost.{name} is required for mode={self.cost.mode!r}.")
        return float(value)

    @staticmethod
    def _diag_from_vector(values: list[float], expected_dim: int, name: str) -> np.ndarray:
        if len(values) != expected_dim:
            raise ValueError(f"{name} must contain {expected_dim} values, got {len(values)}.")
        return np.diag(np.asarray(values, dtype=float))
