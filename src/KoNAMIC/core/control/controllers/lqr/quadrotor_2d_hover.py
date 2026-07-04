from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import expm, solve_discrete_are

from ..base_controller import BaseController


class Quadrotor2DLQRHoverController(BaseController):
    """
    DLQR controller for the planar quadrotor around hover.

    State convention:
        x = [x, z, theta, x_dot, z_dot, theta_dot]

    Input convention:
        u = [T, M]

    Hover equilibrium:
        theta = 0
        x_dot = 0
        z_dot = 0
        theta_dot = 0
        u_eq = [m * g, 0]

    Linearized dynamics around hover:
        x_ddot     ≈ -g * theta
        z_ddot     ≈ delta_T / m
        theta_ddot = M / I_y

    The controller tracks a full state reference trajectory of shape (T, 6):

        u = u_eq - K @ (x - x_ref)

    This controller is intended for small-angle tracking around hover.
    """

    def __init__(
        self,
        *,
        dt: float,
        mass: float,
        inertia_y: float,
        gravity: float,
        q_diag: np.ndarray,
        r_diag: np.ndarray,
        thrust_min: float,
        thrust_max: float,
        moment_min: float,
        moment_max: float,
        max_moment_rate: float | None,
    ) -> None:
        super().__init__(
            dt=dt,
            x_dim=6,
            u_dim=2,
            u_min=np.array([thrust_min, moment_min], dtype=float),
            u_max=np.array([thrust_max, moment_max], dtype=float),
            name="Quadrotor2DLQRHoverController",
        )

        self.mass = float(mass)
        self.inertia_y = float(inertia_y)
        self.gravity = float(gravity)
        self.hover_thrust = self.mass * self.gravity

        self.Q = np.diag(np.asarray(q_diag, dtype=float).reshape(-1))
        self.R = np.diag(np.asarray(r_diag, dtype=float).reshape(-1))

        if self.Q.shape != (self.x_dim, self.x_dim):
            raise ValueError(
                f"q_diag must define a ({self.x_dim}, {self.x_dim}) matrix, "
                f"got Q.shape={self.Q.shape}."
            )

        if self.R.shape != (self.u_dim, self.u_dim):
            raise ValueError(
                f"r_diag must define a ({self.u_dim}, {self.u_dim}) matrix, "
                f"got R.shape={self.R.shape}."
            )

        if thrust_min > self.hover_thrust:
            raise ValueError(
                f"thrust_min={thrust_min} is greater than hover_thrust={self.hover_thrust}."
            )

        if thrust_max < self.hover_thrust:
            raise ValueError(
                f"thrust_max={thrust_max} is smaller than hover_thrust={self.hover_thrust}."
            )

        self.max_moment_rate = None if max_moment_rate is None else float(max_moment_rate)
        self.prev_moment = 0.0

        self.A_d: np.ndarray | None = None
        self.B_d: np.ndarray | None = None
        self.K: np.ndarray | None = None

        self.x_ref_traj: np.ndarray | None = None
        self.step_idx = 0

        self.x_traj: list[np.ndarray] = []
        self.x_ref_k_traj: list[np.ndarray] = []
        self.error_traj: list[np.ndarray] = []
        self.u_unsat_traj: list[np.ndarray] = []

        self.build()

    def reset(self) -> None:
        super().reset()

        self.x_ref_traj = None
        self.step_idx = 0
        self.prev_moment = 0.0

        self.x_traj.clear()
        self.x_ref_k_traj.clear()
        self.error_traj.clear()
        self.u_unsat_traj.clear()

    def set_reference(self, x_ref_traj: np.ndarray) -> None:
        x_ref_traj = np.asarray(x_ref_traj, dtype=float)

        if x_ref_traj.ndim != 2:
            raise ValueError(
                f"x_ref_traj must be a 2D array with shape (T, {self.x_dim}), "
                f"got shape {x_ref_traj.shape}."
            )

        if x_ref_traj.shape[1] != self.x_dim:
            raise ValueError(
                f"x_ref_traj must have shape (T, {self.x_dim}), "
                f"got {x_ref_traj.shape}."
            )

        if x_ref_traj.shape[0] == 0:
            raise ValueError("x_ref_traj must contain at least one time step.")

        self.x_ref_traj = x_ref_traj.copy()
        self.step_idx = 0
        super().set_reference(self.x_ref_traj)

    def set_initial_conditions(self, observation: Any) -> None:
        x_init = self._extract_state_from_observation(observation)
        super().set_initial_conditions(x_init)

        self.step_idx = 0
        self.prev_moment = 0.0

    def build(self) -> None:
        A_c, B_c = self._linearize_hover_continuous()
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
            raise ValueError(f"x_k must have shape ({self.x_dim},), got {x_k.shape}.")

        k = min(self.step_idx, self.x_ref_traj.shape[0] - 1)
        x_ref_k = self.x_ref_traj[k]

        error = x_k - x_ref_k

        u_eq = np.array([self.hover_thrust, 0.0], dtype=float)
        u_unsat = u_eq - self.K @ error

        if self.max_moment_rate is not None:
            u_unsat = self._apply_moment_rate_limit(u_unsat)

        u = self._clip_control(u_unsat)

        self.step_idx += 1

        self.x_traj.append(x_k.copy())
        self.x_ref_k_traj.append(x_ref_k.copy())
        self.error_traj.append(error.copy())
        self.u_unsat_traj.append(u_unsat.copy())

        return self._store_control(
            u,
            info={
                "x_ref_k": x_ref_k.copy(),
                "error": error.copy(),
                "u_unsat": u_unsat.copy(),
                "u_eq": u_eq.copy(),
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

    def _linearize_hover_continuous(self) -> tuple[np.ndarray, np.ndarray]:
        g = self.gravity
        m = self.mass
        Iy = self.inertia_y

        A = np.array(
            [
                [0.0, 0.0,  0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0,  0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0,  0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, -g,   0.0, 0.0, 0.0],
                [0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        )

        B = np.array(
            [
                [0.0,     0.0],
                [0.0,     0.0],
                [0.0,     0.0],
                [0.0,     0.0],
                [1.0 / m, 0.0],
                [0.0,     1.0 / Iy],
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

    def _apply_moment_rate_limit(self, u: np.ndarray) -> np.ndarray:
        u_limited = np.asarray(u, dtype=float).reshape(self.u_dim).copy()

        moment = float(u_limited[1])
        dM = moment - self.prev_moment
        dM_max = self.max_moment_rate * self.dt
        dM = float(np.clip(dM, -dM_max, dM_max))

        moment_limited = self.prev_moment + dM
        u_limited[1] = moment_limited

        self.prev_moment = moment_limited

        return u_limited