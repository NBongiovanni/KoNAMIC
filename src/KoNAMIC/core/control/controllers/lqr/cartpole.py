from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import expm, solve_discrete_are

from ..base_controller import BaseController


class CartPoleLQRController(BaseController):
    """
    DLQR controller for the upright CartPole.

    State convention:
        x = [p, theta, p_dot, theta_dot]

    Input convention:
        u = [force]

    Equilibrium:
        theta = 0
        p_dot = 0
        theta_dot = 0
        u_eq = 0

    The controller tracks a full state reference trajectory of shape (T, 4).
    In practice, the most important components are usually:
        p_ref = reference[:, 0]
        theta_ref = 0
    """

    def __init__(
        self,
        *,
        dt: float,
        cart_mass: float,
        pole_mass: float,
        pole_length: float,
        gravity: float,
        q_diag: np.ndarray,
        r_diag: np.ndarray,
        force_min: float,
        force_max: float,
    ) -> None:
        super().__init__(
            dt=dt,
            x_dim=4,
            u_dim=1,
            u_min=np.array([force_min], dtype=float),
            u_max=np.array([force_max], dtype=float),
            name="CartPoleLQRController",
        )

        self.cart_mass = float(cart_mass)
        self.pole_mass = float(pole_mass)
        self.pole_length = float(pole_length)
        self.gravity = float(gravity)

        self.Q = np.diag(np.asarray(q_diag, dtype=float).reshape(-1))
        self.R = np.diag(np.asarray(r_diag, dtype=float).reshape(-1))

        if self.Q.shape != (self.x_dim, self.x_dim):
            raise ValueError(
                f"q_diag must define a ({self.x_dim}, {self.x_dim}) matrix, "
                f"got Q.shape={self.Q.shape}"
            )

        if self.R.shape != (self.u_dim, self.u_dim):
            raise ValueError(
                f"r_diag must define a ({self.u_dim}, {self.u_dim}) matrix, "
                f"got R.shape={self.R.shape}"
            )

        self.A_d: np.ndarray | None = None
        self.B_d: np.ndarray | None = None
        self.K: np.ndarray | None = None

        self.x_ref_traj: np.ndarray | None = None
        self.step_idx = 0

        self.x_traj: list[np.ndarray] = []
        self.x_ref_k_traj: list[np.ndarray] = []
        self.error_traj: list[np.ndarray] = []

        self.build()

    def reset(self) -> None:
        super().reset()
        self.x_ref_traj = None
        self.step_idx = 0
        self.x_traj.clear()
        self.x_ref_k_traj.clear()
        self.error_traj.clear()

    def set_reference(self, x_ref_traj: np.ndarray) -> None:
        x_ref_traj = np.asarray(x_ref_traj, dtype=float)

        if x_ref_traj.ndim != 2:
            raise ValueError(
                f"x_ref_traj must be a 2D array with shape (T, {self.x_dim}), "
                f"got shape {x_ref_traj.shape}"
            )

        if x_ref_traj.shape[1] != self.x_dim:
            raise ValueError(
                f"x_ref_traj must have shape (T, {self.x_dim}), "
                f"got {x_ref_traj.shape}"
            )

        if x_ref_traj.shape[0] == 0:
            raise ValueError("x_ref_traj must contain at least one time step.")

        self.x_ref_traj = x_ref_traj.copy()
        self.step_idx = 0
        super().set_reference(x_ref_traj)

    def set_initial_conditions(self, observation: Any) -> None:
        x_init = self._extract_state_from_observation(observation)
        super().set_initial_conditions(x_init)
        self.step_idx = 0

    def build(self) -> None:
        A_c, B_c = self._linearize_upright_continuous()
        A_d, B_d = self._discretize_zero_order_hold(A_c=A_c, B_c=B_c)

        P = solve_discrete_are(A_d, B_d, self.Q, self.R)
        K = np.linalg.solve(B_d.T @ P @ B_d + self.R, B_d.T @ P @ A_d)

        self.A_d = A_d
        self.B_d = B_d
        self.K = K
        self.is_built = True

    def compute_control(self, observation: Any) -> np.ndarray:
        if self.K is None:
            raise RuntimeError("Controller must be built before compute_control().")

        if self.x_ref_traj is None:
            raise RuntimeError("Call set_reference() before compute_control().")

        x_k = self._extract_state_from_observation(observation)

        if x_k.shape != (self.x_dim,):
            raise ValueError(f"x_k must have shape ({self.x_dim},), got {x_k.shape}")

        k = min(self.step_idx, self.x_ref_traj.shape[0] - 1)
        x_ref_k = self.x_ref_traj[k]

        error = x_k - x_ref_k
        u = -self.K @ error
        u = self._clip_control(u)

        self.step_idx += 1

        self.x_traj.append(x_k.copy())
        self.x_ref_k_traj.append(x_ref_k.copy())
        self.error_traj.append(error.copy())

        return self._store_control(
            u,
            info={
                "x_ref_k": x_ref_k.copy(),
                "error": error.copy(),
                "K": self.K.copy(),
            },
        )

    def _extract_state_from_observation(self, observation: Any) -> np.ndarray:
        if isinstance(observation, dict):
            if "x_k" not in observation:
                raise KeyError("Observation dict must contain key 'x_k'.")
            x_k = observation["x_k"]
        else:
            x_k = observation

        return np.asarray(x_k, dtype=float).reshape(-1)

    def _linearize_upright_continuous(self) -> tuple[np.ndarray, np.ndarray]:
        M = self.cart_mass
        m = self.pole_mass
        l = self.pole_length
        g = self.gravity

        A = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, -m * g / M, 0.0, 0.0],
                [0.0, (M + m) * g / (M * l), 0.0, 0.0],
            ],
            dtype=float,
        )

        B = np.array(
            [
                [0.0],
                [0.0],
                [1.0 / M],
                [1.0 / (M * l)],
            ],
            dtype=float,
        )

        return A, B

    def _discretize_zero_order_hold(
        self,
        *,
        A_c: np.ndarray,
        B_c: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_x = A_c.shape[0]
        n_u = B_c.shape[1]

        augmented = np.zeros((n_x + n_u, n_x + n_u), dtype=float)
        augmented[:n_x, :n_x] = A_c
        augmented[:n_x, n_x:] = B_c

        exp_augmented = expm(augmented * self.dt)

        A_d = exp_augmented[:n_x, :n_x]
        B_d = exp_augmented[:n_x, n_x:]

        return A_d, B_d