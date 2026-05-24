from __future__ import annotations

from typing import Optional, Any

import numpy as np

from .base_controller import BaseController


class PIDController(BaseController):
    """
    PID trajectoriel.

    - set_reference(...) attend une trajectoire complète de forme (T, x_dim)
    - le PID ne suit que les composantes d'état indiquées par tracked_state_indices
    - à chaque appel de compute_control(...), la référence courante est x_ref_traj[k]
    - compute_control(...) accepte maintenant une observation générique
    """

    def __init__(
        self,
        dt: float,
        x_dim: int,
        u_dim: int,
        tracked_state_indices: list[int] | np.ndarray,
        kp: np.ndarray,
        ki: np.ndarray,
        kd: np.ndarray,
        u_min: Optional[np.ndarray] = None,
        u_max: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(
            dt=dt,
            x_dim=x_dim,
            u_dim=u_dim,
            u_min=u_min,
            u_max=u_max,
            name="PIDController",
        )

        self.tracked_state_indices = np.asarray(tracked_state_indices, dtype=int).reshape(-1)
        if self.tracked_state_indices.ndim != 1 or self.tracked_state_indices.size == 0:
            raise ValueError("tracked_state_indices must be a non-empty 1D array-like.")

        if np.any(self.tracked_state_indices < 0) or np.any(self.tracked_state_indices >= self.x_dim):
            raise ValueError(
                f"tracked_state_indices must be in [0, {self.x_dim - 1}], "
                f"got {self.tracked_state_indices}"
            )

        self.n_tracked = int(self.tracked_state_indices.size)

        self.kp = np.asarray(kp, dtype=float).reshape(-1)
        self.ki = np.asarray(ki, dtype=float).reshape(-1)
        self.kd = np.asarray(kd, dtype=float).reshape(-1)

        if self.kp.shape != (self.n_tracked,):
            raise ValueError(f"kp must have shape ({self.n_tracked},), got {self.kp.shape}")
        if self.ki.shape != (self.n_tracked,):
            raise ValueError(f"ki must have shape ({self.n_tracked},), got {self.ki.shape}")
        if self.kd.shape != (self.n_tracked,):
            raise ValueError(f"kd must have shape ({self.n_tracked},), got {self.kd.shape}")

        self.x_ref_traj: Optional[np.ndarray] = None
        self.u_eq: Optional[np.ndarray] = None
        self.step_idx: int = 0

        self.integral_error: Optional[np.ndarray] = None
        self.prev_error: Optional[np.ndarray] = None

        self.x_traj: list[np.ndarray] = []
        self.error_traj: list[np.ndarray] = []
        self.x_ref_k_traj: list[np.ndarray] = []

    def reset(self) -> None:
        super().reset()
        self.x_ref_traj = None
        self.u_eq = None
        self.step_idx = 0
        self.integral_error = None
        self.prev_error = None
        self.x_traj.clear()
        self.error_traj.clear()
        self.x_ref_k_traj.clear()

    def set_reference(self, x_ref_traj: np.ndarray) -> None:
        x_ref_traj = np.asarray(x_ref_traj, dtype=float)

        if x_ref_traj.ndim != 2:
            raise ValueError(
                f"x_ref_traj must be a 2D array of shape (T, {self.x_dim}), "
                f"got ndim={x_ref_traj.ndim}, shape={x_ref_traj.shape}"
            )

        if x_ref_traj.shape[1] != self.x_dim:
            raise ValueError(
                f"x_ref_traj must have shape (T, {self.x_dim}), got {x_ref_traj.shape}"
            )

        if x_ref_traj.shape[0] == 0:
            raise ValueError("x_ref_traj must contain at least one time step.")

        self.x_ref_traj = x_ref_traj
        self.step_idx = 0
        super().set_reference(x_ref_traj)

    def set_operating_point(self, u_eq: np.ndarray) -> None:
        u_eq = np.asarray(u_eq, dtype=float).reshape(-1)
        if u_eq.shape != (self.u_dim,):
            raise ValueError(f"u_eq must have shape ({self.u_dim},), got {u_eq.shape}")
        self.u_eq = u_eq

    def set_initial_conditions(self, observation: Any) -> None:
        x_init = self._extract_state_from_observation(observation)
        super().set_initial_conditions(x_init)

        self.step_idx = 0
        self.integral_error = np.zeros(self.n_tracked, dtype=float)
        self.prev_error = np.zeros(self.n_tracked, dtype=float)

    def build(self) -> None:
        self.is_built = True

    def compute_control(self, observation: Any) -> np.ndarray:
        x_k = self._extract_state_from_observation(observation)

        if x_k.shape != (self.x_dim,):
            raise ValueError(f"x_k must have shape ({self.x_dim},), got {x_k.shape}")

        k = min(self.step_idx, self.x_ref_traj.shape[0] - 1)
        x_ref_k: np.ndarray = self.x_ref_traj[k]

        error = self._compute_error(x_k, x_ref_k)

        if self.integral_error is None:
            self.integral_error = np.zeros(self.n_tracked, dtype=float)
        if self.prev_error is None:
            self.prev_error = np.zeros(self.n_tracked, dtype=float)

        self.integral_error = self._update_integral(error, self.integral_error)
        derivative_error = self._compute_derivative(error, self.prev_error)

        u = self._compute_pid_control(
            x_k=x_k,
            x_ref_k=x_ref_k,
            error=error,
            integral_error=self.integral_error,
            derivative_error=derivative_error,
        )

        if self.u_eq is not None:
            u = np.asarray(u, dtype=float).reshape(-1) + self.u_eq

        # u = self._clip_control(u)
        self.prev_error = error.copy()
        self.step_idx += 1

        self.x_traj.append(x_k.copy())
        self.x_ref_k_traj.append(x_ref_k.copy())
        self.error_traj.append(error.copy())

        return self._store_control(
            u,
            info={
                "x_ref_k": x_ref_k.copy(),
                "tracked_state_indices": self.tracked_state_indices.copy(),
                "error": error.copy(),
                "integral_error": self.integral_error.copy(),
                "derivative_error": derivative_error.copy(),
            },
        )

    def _extract_state_from_observation(self, observation: Any) -> np.ndarray:
        """
        Accepte :
        - directement un np.ndarray représentant x_k
        - ou un dict contenant la clé 'x_k'
        """
        if isinstance(observation, dict):
            if "x_k" not in observation:
                raise KeyError("Observation dict must contain key 'x_k'.")
            x_k = observation["x_k"]
        else:
            x_k = observation
        return np.asarray(x_k, dtype=float).reshape(-1)

    def _compute_error(self, x_k: np.ndarray, x_ref_k: np.ndarray) -> np.ndarray:
        x_k_tracked = x_k[self.tracked_state_indices]
        x_ref_k_tracked = x_ref_k[self.tracked_state_indices]
        return x_k_tracked - x_ref_k_tracked

    def _update_integral(self, error: np.ndarray, integral_error: np.ndarray) -> np.ndarray:
        return integral_error + self.dt * error

    def _compute_derivative(self, error: np.ndarray, prev_error: np.ndarray) -> np.ndarray:
        return (error - prev_error) / self.dt

    def _compute_pid_control(
        self,
        x_k: np.ndarray,
        x_ref_k: np.ndarray,
        error: np.ndarray,
        integral_error: np.ndarray,
        derivative_error: np.ndarray,
    ) -> np.ndarray:
        pid_terms = (
            self.kp * error
            + self.ki * integral_error
            + self.kd * derivative_error
        )

        u = np.zeros(self.u_dim, dtype=float)
        n_map = min(self.n_tracked, self.u_dim)
        u[:n_map] = -pid_terms[:n_map]
        return u