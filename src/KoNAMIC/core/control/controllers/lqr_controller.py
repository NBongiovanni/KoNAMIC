from __future__ import annotations

from typing import Optional

import numpy as np

from .base_controller import BaseController


class LQRController(BaseController):
    """
    Squelette pour un contrôleur LQR / DLQR.

    La partie mathématique (calcul de K, linéarisation, équilibre, etc.)
    est volontairement laissée hors de ce squelette.
    """

    def __init__(
        self,
        dt: float,
        x_dim: int,
        u_dim: int,
        u_min: Optional[np.ndarray] = None,
        u_max: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(
            dt=dt,
            x_dim=x_dim,
            u_dim=u_dim,
            u_min=u_min,
            u_max=u_max,
            name="LQRController",
        )

        self.x_ref: Optional[np.ndarray] = None
        self.u_eq: Optional[np.ndarray] = None
        self.K: Optional[np.ndarray] = None

        self.x_traj: list[np.ndarray] = []
        self.error_traj: list[np.ndarray] = []

    def reset(self) -> None:
        super().reset()
        self.x_traj.clear()
        self.error_traj.clear()

    def set_reference(self, x_ref: np.ndarray) -> None:
        x_ref = np.asarray(x_ref, dtype=float).reshape(-1)
        if x_ref.shape != (self.x_dim,):
            raise ValueError(f"x_ref must have shape ({self.x_dim},), got {x_ref.shape}")

        self.x_ref = x_ref
        super().set_reference(x_ref)

    def set_operating_point(self, u_eq: np.ndarray) -> None:
        u_eq = np.asarray(u_eq, dtype=float).reshape(-1)
        if u_eq.shape != (self.u_dim,):
            raise ValueError(f"u_eq must have shape ({self.u_dim},), got {u_eq.shape}")
        self.u_eq = u_eq

    def build(self) -> None:
        self.K = self._compute_lqr_gain()
        self.is_built = True

    def compute_control(self, x_k: np.ndarray) -> np.ndarray:
        if self.K is None:
            raise RuntimeError("Call build() before compute_control().")
        if self.x_ref is None:
            raise RuntimeError("Call set_reference() before compute_control().")

        x_k = np.asarray(x_k, dtype=float).reshape(-1)
        if x_k.shape != (self.x_dim,):
            raise ValueError(f"x_k must have shape ({self.x_dim},), got {x_k.shape}")

        error = x_k - self.x_ref
        u = self._compute_lqr_control(x_k=x_k, error=error)

        if self.u_eq is not None:
            u = np.asarray(u, dtype=float).reshape(-1) + self.u_eq

        u = self._clip_control(u)

        self.x_traj.append(x_k.copy())
        self.error_traj.append(error.copy())
        return self._store_control(u, info={"error": error})

    def _compute_lqr_gain(self) -> np.ndarray:
        raise NotImplementedError("TODO: compute the LQR gain K")

    def _compute_lqr_control(self, x_k: np.ndarray, error: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: compute u = -K e")