from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .mpc_controller_base import MPCControllerBase


class FullStateNMPCController(MPCControllerBase):
    """
    NMPC avec connaissance complète du modèle / de l'état.

    Le MPC travaille directement dans l'espace d'état x.
    """

    def __init__(
        self,
        dt: float,
        x_dim: int,
        u_dim: int,
        horizon: int,
        solver_backend,
        control_runs_dir: Optional[str] = None,
        u_min: Optional[np.ndarray] = None,
        u_max: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(
            dt=dt,
            x_dim=x_dim,
            u_dim=u_dim,
            prediction_dim=x_dim,
            horizon=horizon,
            solver_backend=solver_backend,
            control_runs_dir=control_runs_dir,
            u_min=u_min,
            u_max=u_max,
            name="FullStateNMPCController",
        )

        self.x_ref_traj: Optional[np.ndarray] = None
        self.x_traj: list[np.ndarray] = []

    def reset(self) -> None:
        super().reset()
        self.x_ref_traj = None
        self.x_traj.clear()

    def set_reference(self, x_reference: np.ndarray) -> None:
        self.x_ref_traj = np.asarray(x_reference, dtype=float)
        super().set_reference(self.x_ref_traj)

    def _build_cost_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # TODO: remplacer par votre vraie construction Q / Qf / R
        Q = np.eye(self.x_dim)
        Qf = np.eye(self.x_dim)
        R = np.eye(self.u_dim)
        return Q, Qf, R

    def _build_prediction_dynamics(self):
        """
        Retourne la dynamique discrète utilisée par le solveur NMPC.
        """
        raise NotImplementedError("TODO: build the full-state prediction dynamics")

    def _observation_to_prediction_state(self, observation: Any) -> np.ndarray:
        x_k = np.asarray(observation, dtype=float).reshape(-1)
        if x_k.shape != (self.x_dim,):
            raise ValueError(f"State must have shape ({self.x_dim},), got {x_k.shape}")

        self.x_traj.append(x_k.copy())
        return x_k