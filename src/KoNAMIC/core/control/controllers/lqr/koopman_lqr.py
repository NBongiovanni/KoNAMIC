from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.linalg import solve_discrete_are

from KoNAMIC.core.models.base_koop_model import BaseKoopModel
from KoNAMIC.core.models.model_config import ModelConfig, SensorAutoEncoderConfig
from KoNAMIC.core.scaling import DatasetScalers

from ..base_controller import BaseController


class KoopmanLQRController(BaseController):
    """
    DLQR controller applied to a learned linear Koopman representation.

    First version: stabilization around the origin. The learned dynamics are

        z[k+1] = A z[k] + B u[k]

    where the first x_dim components of z are assumed to be the physical state x.
    The LQR cost penalizes only those physical components:

        Qz = Cx.T @ Qx @ Cx,  Cx = [I_x 0]

    The control law is computed in the scaled input space used by the Koopman
    model, then inverse-transformed to physical units when an input scaler is
    available.
    """

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        koop_model: BaseKoopModel,
        scalers: DatasetScalers,
        q_diag: np.ndarray,
        r_diag: np.ndarray,
        u_min: np.ndarray | None = None,
        u_max: np.ndarray | None = None,
    ) -> None:
        if model_config.z_dynamics.model != "linear":
            raise ValueError(
                "KoopmanLQRController only supports linear Koopman dynamics. "
                f"Got model.z_dynamics.model={model_config.z_dynamics.model!r}."
            )

        if not isinstance(model_config.auto_encoder, SensorAutoEncoderConfig):
            raise ValueError("KoopmanLQRController currently requires a sensor model config.")

        if not model_config.auto_encoder.include_state_in_z:
            raise ValueError(
                "KoopmanLQRController requires model.auto_encoder.include_state_in_z=True "
                "so that the first z components are the physical state."
            )

        if model_config.z_dynamics.x_dim is None:
            raise ValueError("model.z_dynamics.x_dim must be set before building KoopmanLQRController.")

        self.model_config = model_config
        self.koop_model = koop_model
        self.scalers = scalers
        self.x_scaler = scalers.x
        self.u_scaler = scalers.u
        self.device = next(self.koop_model.parameters()).device

        self.z_dim = int(model_config.z_dynamics.z_dim)
        x_dim = int(model_config.z_dynamics.x_dim)
        u_dim = int(koop_model.u_dim)

        super().__init__(
            dt=float(model_config.dt),
            x_dim=x_dim,
            u_dim=u_dim,
            u_min=u_min,
            u_max=u_max,
            name="KoopmanLQRController",
        )

        q_diag = np.asarray(q_diag, dtype=float).reshape(-1)
        r_diag = np.asarray(r_diag, dtype=float).reshape(-1)

        if q_diag.shape != (self.x_dim,):
            raise ValueError(f"q_diag must have shape ({self.x_dim},), got {q_diag.shape}.")
        if r_diag.shape != (self.u_dim,):
            raise ValueError(f"r_diag must have shape ({self.u_dim},), got {r_diag.shape}.")

        self.Qx = np.diag(q_diag)
        self.R = np.diag(r_diag)
        self.Cx = np.zeros((self.x_dim, self.z_dim), dtype=float)
        self.Cx[:, : self.x_dim] = np.eye(self.x_dim, dtype=float)
        self.Qz = self.Cx.T @ self.Qx @ self.Cx

        self.A: np.ndarray | None = None
        self.B: np.ndarray | None = None
        self.K: np.ndarray | None = None
        self.closed_loop_eigs: np.ndarray | None = None

        self.z_traj: list[np.ndarray] = []
        self.x_scaled_traj: list[np.ndarray] = []
        self.u_scaled_traj: list[np.ndarray] = []
        self.u_unsat_traj: list[np.ndarray] = []
        self.x_ref_traj: np.ndarray | None = None
        self.z_ref_traj: np.ndarray | None = None

        self.build()

    def reset(self) -> None:
        super().reset()
        self.z_traj.clear()
        self.x_scaled_traj.clear()
        self.u_scaled_traj.clear()
        self.u_unsat_traj.clear()
        self.x_ref_traj = None
        self.z_ref_traj = None

    def set_reference(self, reference: Any) -> None:
        super().set_reference(reference)

        if reference is None:
            self.x_ref_traj = None
            self.z_ref_traj = None
            return

        ref = np.asarray(reference, dtype=float)
        if ref.ndim == 1 and ref.shape[0] == self.x_dim:
            self.x_ref_traj = ref.reshape(1, -1).copy()
            self.z_ref_traj = None
            return

        if ref.ndim == 2 and ref.shape[1] == self.x_dim:
            self.x_ref_traj = ref.copy()
            self.z_ref_traj = None
            return

        if ref.ndim == 1 and ref.shape[0] == self.z_dim:
            self.z_ref_traj = ref.reshape(1, -1).copy()
            self.x_ref_traj = None
            return

        if ref.ndim == 2 and ref.shape[1] == self.z_dim:
            self.z_ref_traj = ref.copy()
            self.x_ref_traj = None
            return

        self.x_ref_traj = None
        self.z_ref_traj = None

    def set_initial_conditions(self, observation: Any) -> None:
        x_init = self._extract_state_from_observation(observation)
        super().set_initial_conditions(x_init)

    def build(self) -> None:
        A, B = self._get_linear_koopman_matrices()

        if A.shape != (self.z_dim, self.z_dim):
            raise ValueError(f"A must have shape ({self.z_dim}, {self.z_dim}), got {A.shape}.")
        if B.shape != (self.z_dim, self.u_dim):
            raise ValueError(f"B must have shape ({self.z_dim}, {self.u_dim}), got {B.shape}.")

        P = solve_discrete_are(A, B, self.Qz, self.R)
        K = np.linalg.solve(B.T @ P @ B + self.R, B.T @ P @ A)

        self.A = A
        self.B = B
        self.K = K
        self.closed_loop_eigs = np.linalg.eigvals(A - B @ K)
        self.is_built = True

    def compute_control(self, observation: Any) -> np.ndarray:
        if self.K is None:
            raise RuntimeError("Controller must be built before compute_control().")

        z_k = self._encode_observation_to_z(observation)
        u_scaled = -self.K @ z_k
        u_physical_unsat = self._inverse_scale_control(u_scaled)
        u_physical = self._clip_control(u_physical_unsat)

        x_scaled = z_k[: self.x_dim]
        self.z_traj.append(z_k.copy())
        self.x_scaled_traj.append(x_scaled.copy())
        self.u_scaled_traj.append(u_scaled.copy())
        self.u_unsat_traj.append(u_physical_unsat.copy())

        return self._store_control(
            u_physical,
            info={
                "z_k": z_k.copy(),
                "x_scaled_k": x_scaled.copy(),
                "u_scaled": u_scaled.copy(),
                "u_unsat": u_physical_unsat.copy(),
                "K": self.K.copy(),
            },
        )

    @torch.no_grad()
    def _get_linear_koopman_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        A_t, B_t = self.koop_model.construct_koop_matrices()
        A = A_t.detach().cpu().numpy().astype(float)
        B = B_t.detach().cpu().numpy().astype(float)
        return A, B

    def _encode_observation_to_z(self, observation: Any) -> np.ndarray:
        if isinstance(observation, dict) and "z_k" in observation:
            z_k = np.asarray(observation["z_k"], dtype=float).reshape(-1)
            if z_k.shape != (self.z_dim,):
                raise ValueError(f"z_k must have shape ({self.z_dim},), got {z_k.shape}.")
            return z_k

        x_k = self._extract_state_from_observation(observation)
        if self.x_scaler is None:
            x_scaled = x_k.reshape(1, -1).astype(np.float32)
        else:
            x_scaled = self.x_scaler.transform(x_k.reshape(1, -1)).astype(np.float32)
        x_t = torch.as_tensor(x_scaled, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            z_t = self.koop_model.project(x_t)

        z_k = z_t.detach().cpu().numpy().reshape(-1).astype(float)
        if z_k.shape != (self.z_dim,):
            raise ValueError(f"Encoded latent state must have shape ({self.z_dim},), got {z_k.shape}.")
        return z_k

    def _extract_state_from_observation(self, observation: Any) -> np.ndarray:
        if isinstance(observation, dict):
            if "x_k" not in observation:
                raise KeyError("Observation dict must contain key 'x_k' or 'z_k'.")
            x_k = observation["x_k"]
        else:
            x_k = observation

        x_k = np.asarray(x_k, dtype=float).reshape(-1)
        if x_k.shape != (self.x_dim,):
            raise ValueError(f"x_k must have shape ({self.x_dim},), got {x_k.shape}.")
        return x_k

    def _inverse_scale_control(self, u_scaled: np.ndarray) -> np.ndarray:
        u_scaled = np.asarray(u_scaled, dtype=float).reshape(-1)
        if u_scaled.shape != (self.u_dim,):
            raise ValueError(f"u_scaled must have shape ({self.u_dim},), got {u_scaled.shape}.")

        if self.u_scaler is None:
            return u_scaled

        return self.u_scaler.inverse_transform(u_scaled.reshape(1, -1))[0]
