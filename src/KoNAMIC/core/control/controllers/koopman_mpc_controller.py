from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

from KoNAMIC.core.models import BaseKoopModel

from .mpc_controller_base import MPCControllerBase
from ..dynamics.casadi_dynamics import build_latent_dynamics_function


class KoopmanMPCController(MPCControllerBase):
    """
    Contrôleur MPC basé sur un modèle de Koopman appris.

    Cette classe contient :
    - la logique spécifique au latent z
    - la projection observation -> z
    - la construction de la dynamique latente
    - le post-traitement éventuel via scaler de commande
    """

    def __init__(
        self,
        model_params: dict,
        control_params: dict,
        solver_backend,
        koop_model: BaseKoopModel,
        u_scaler: StandardScaler,
        x_scaler: Optional[StandardScaler] = None,
    ) -> None:

        self.model_params = model_params
        self.control_params = control_params
        self.koop_model = koop_model
        self.u_scaler = u_scaler
        self.x_scaler = x_scaler

        self.device = next(self.koop_model.parameters()).device
        self.z_dim = int(self.model_params["z_dynamics"]["z_dim"])
        self.x_in_z = bool(self.model_params["auto_encoder"]["include_state_in_z"])

        u_min, u_max = self._get_scaled_input_bounds()

        super().__init__(
            dt=float(control_params["dt"]),
            x_dim=None,
            u_dim=int(self.koop_model.u_dim),
            prediction_dim=self.z_dim,
            horizon=int(control_params["num_steps_horizon"]),
            solver_backend=solver_backend,
            control_runs_dir=control_params.get("control_runs_dir"),
            u_min=u_min,
            u_max=u_max,
            name="KoopmanMPCController",
        )
        self.z_ref_traj: Optional[np.ndarray] = None
        self.x_ref_traj = None
        self.u_scaled_traj: list[np.ndarray] = []
        self.u_physical_traj: list[np.ndarray] = []

    def reset(self) -> None:
        super().reset()
        self.z_ref_traj = None
        self.u_scaled_traj.clear()
        self.u_physical_traj.clear()

    def set_reference(self, z_reference: np.ndarray) -> None:
        self.z_ref_traj = np.asarray(z_reference, dtype=float)
        super().set_reference(self.z_ref_traj)

    def _build_cost_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # TODO: remplacer par votre vraie construction Q / Qf / R
        Q = np.eye(self.z_dim)
        Qf = np.eye(self.z_dim)
        R = np.eye(self.u_dim)
        return Q, Qf, R

    def _build_input_rate_matrix(self) -> np.ndarray:
        if "cost" in self.control_params and "S" in self.control_params["cost"]:
            return np.diag(self.control_params["cost"]["S"])
        return super()._build_input_rate_matrix()

    def _build_prediction_dynamics(self):
        return build_latent_dynamics_function(
            z_dynamics_model=self.model_params["z_dynamics"]["model"],
            z_dim=self.z_dim,
            u_dim=self.u_dim,
            koop_model=self.koop_model,
            augment_actuated=self.model_params["z_dynamics"]["affine_term"],
        )

    def _observation_to_prediction_state(self, observation: Any) -> np.ndarray:
        """
        Ici, 'observation' peut être :
        - un état x_k
        - un dictionnaire contenant historique / entrée précédente
        - toute structure adaptée à votre encodeur

        On délègue la logique à `_encode_observation_to_z`.
        """
        z_k = self._encode_observation_to_z(observation)
        z_k = np.asarray(z_k, dtype=float).reshape(-1)

        if z_k.shape != (self.z_dim,):
            raise ValueError(f"Encoded latent state must have shape ({self.z_dim},), got {z_k.shape}")
        return z_k

    def _postprocess_control(self, u_raw: np.ndarray) -> np.ndarray:
        u_scaled = np.asarray(u_raw, dtype=float).reshape(-1)
        if u_scaled.shape != (self.u_dim,):
            raise ValueError(f"Scaled control must have shape ({self.u_dim},), got {u_scaled.shape}")

        self.u_scaled_traj.append(u_scaled.copy())

        u_physical = self.u_scaler.inverse_transform(u_scaled.reshape(1, -1))[0]
        u_physical = self._clip_control(u_physical)
        self.u_physical_traj.append(u_physical.copy())
        return u_physical

    def _get_initial_control_guess(self) -> np.ndarray:
        # TODO: remplacer par votre vrai hover guess si besoin
        return np.zeros(self.u_dim, dtype=float)

    def _get_scaled_input_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        constraints = self.control_params["constraints"]
        force_limits = constraints["force_limits"]
        torque_limits = constraints["torque_limits"]

        if self.koop_model.u_dim == 2:
            u_min = np.array([force_limits[0], torque_limits[0]], dtype=float)
            u_max = np.array([force_limits[1], torque_limits[1]], dtype=float)
        elif self.koop_model.u_dim == 4:
            u_min = np.array(
                [force_limits[0], torque_limits[0], torque_limits[0], torque_limits[0]],
                dtype=float,
            )
            u_max = np.array(
                [force_limits[1], torque_limits[1], torque_limits[1], torque_limits[1]],
                dtype=float,
            )
        else:
            raise ValueError(f"Unsupported u_dim={self.koop_model.u_dim}")

        u_min_scaled = self.u_scaler.transform(u_min.reshape(1, -1))[0]
        u_max_scaled = self.u_scaler.transform(u_max.reshape(1, -1))[0]
        return u_min_scaled, u_max_scaled

    def _encode_observation_to_z(self, observation: Any) -> np.ndarray:
        if isinstance(observation, dict):
            x_k = np.asarray(observation["x_k"], dtype=float).reshape(1, -1)
        else:
            x_k = np.asarray(observation, dtype=float).reshape(1, -1)

        if self.x_scaler is not None:
            x_k = self.x_scaler.transform(x_k)

        x_k = x_k.astype(np.float32)

        import torch
        x_t = torch.as_tensor(x_k, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            z_k = self.koop_model.project(x_t)

        return z_k.detach().cpu().numpy().reshape(-1)