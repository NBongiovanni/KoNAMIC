from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from KoNAMIC import config
from KoNAMIC.core.control.config import KmpcControllerConfig
from KoNAMIC.core.control.controllers.mpc.mpc_controller_base import MPCControllerBase
from KoNAMIC.core.control.mpc_solver import AcadosMPCSolver
from KoNAMIC.koopman.models import ModelConfig, BaseKoopModel
from KoNAMIC.core.scaling import DatasetScalers
from KoNAMIC.core.systems import create_system
from .build_latent_dynamics_function import build_latent_dynamics_function
from .kmpc_weight_builder import KmpcWeightBuilder, KmpcWeightMatrices


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
        modality: config.Modality,
        model_config: ModelConfig,
        controller_config: KmpcControllerConfig,
        control_run_dir: Path,
        solver: AcadosMPCSolver,
        koop_model: BaseKoopModel,
        scalers: DatasetScalers,
    ) -> None:
        self.modality = modality
        self.model_config = model_config
        self.controller_config = controller_config
        self.koop_model = koop_model
        self.x_scaler = scalers.x
        self.u_scaler = scalers.u
        self.control_run_dir = control_run_dir

        self.device = next(self.koop_model.parameters()).device
        self.z_dim = self.model_config.z_dynamics.z_dim
        self.x_in_z = self.model_config.auto_encoder.include_state_in_z

        u_min, u_max = self._get_scaled_input_bounds()

        super().__init__(
            dt=float(controller_config.dt),
            x_dim=None,
            u_dim=int(self.koop_model.u_dim),
            prediction_dim=self.z_dim,
            horizon=int(controller_config.num_steps_horizon),
            solver=solver,
            u_min=u_min,
            u_max=u_max,
            name="KoopmanMPCController",
        )
        self.z_ref_traj: Optional[np.ndarray] = None
        self.x_ref_traj = None
        self.u_scaled_traj: list[np.ndarray] = []
        self.u_physical_traj: list[np.ndarray] = []
        self._weight_matrices: KmpcWeightMatrices | None = None

    def reset(self) -> None:
        super().reset()
        self.z_ref_traj = None
        self.u_scaled_traj.clear()
        self.u_physical_traj.clear()

    def set_reference(self, x_reference: np.ndarray) -> None:
        x_ref_traj = np.asarray(x_reference, dtype=float)

        if x_ref_traj.ndim != 2:
            raise ValueError(
                f"x_reference must be 2D, got shape {x_ref_traj.shape}."
            )

        self.x_ref_traj = x_ref_traj.copy()

        if self.modality is config.Modality.SENSOR:
            x_scaled = self.x_scaler.transform(x_ref_traj).astype(np.float32)

            x_t = torch.as_tensor(
                x_scaled,
                dtype=torch.float32,
                device=self.device,
            )

            with torch.no_grad():
                z_t = self.koop_model.project(x_t)

            z_ref_traj = z_t.detach().cpu().numpy()

        elif self.modality is config.Modality.VISION:
            raise NotImplementedError(
                "KMPC vision reference projection is not implemented yet."
            )

        else:
            raise ValueError(f"Unsupported modality: {self.modality.key}")

        self.z_ref_traj = z_ref_traj
        super().set_reference(z_ref_traj)

    def _build_cost_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        weights = self._build_weight_matrices()
        return weights.Q, weights.Qf, weights.R

    def _build_input_rate_matrix(self) -> np.ndarray:
        return self._build_weight_matrices().S

    def _build_weight_matrices(self) -> KmpcWeightMatrices:
        if self._weight_matrices is None:
            builder = KmpcWeightBuilder(
                cost=self.controller_config.cost,
                system_spec=create_system(self.controller_config.system_name),
                z_dim=self.z_dim,
                u_dim=self.u_dim,
                x_in_z=self.x_in_z,
            )
            self._weight_matrices = builder.build()
        return self._weight_matrices

    def _build_prediction_dynamics(self):
        return build_latent_dynamics_function(
            z_dynamics_model=self.model_config.z_dynamics.model,
            z_dim=self.z_dim,
            u_dim=self.u_dim,
            koop_model=self.koop_model,
            augment_actuated=self.model_config.z_dynamics.affine_term,
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
        force_limits = self.controller_config.constraints.force_limits
        torque_limits = self.controller_config.constraints.torque_limits

        if torque_limits is None:
            raise ValueError("controller.constraints.torque_limits is required for KMPC.")

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
