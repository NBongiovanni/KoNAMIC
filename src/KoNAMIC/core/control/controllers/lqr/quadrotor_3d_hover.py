from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import expm, solve_discrete_are

from ..base_controller import BaseController


class Quadrotor3DLQRHoverController(BaseController):
    """
    DLQR controller for the 3D quadrotor around hover.

    State convention:
        x = [
            px, py, pz,
            phi, theta, psi,
            vx, vy, vz,
            p, q, r,
        ]

    Input convention:
        u = [T, Mx, My, Mz]

    Hover equilibrium:
        phi = theta = psi = 0
        vx = vy = vz = 0
        p = q = r = 0
        u_eq = [m * g, 0, 0, 0]

    Linearized dynamics around hover:
        px_dot = vx
        py_dot = vy
        pz_dot = vz

        phi_dot   = p
        theta_dot = q
        psi_dot   = r

        vx_dot ≈  g * theta
        vy_dot ≈ -g * phi
        vz_dot ≈  delta_T / m

        p_dot = Mx / Ix
        q_dot = My / Iy
        r_dot = Mz / Iz

    The controller tracks a full state reference trajectory of shape (T, 12):

        u = u_eq - K @ (x - x_ref)

    This controller is intended for small-angle tracking around hover.
    """

    def __init__(
        self,
        *,
        dt: float,
        mass: float,
        inertia: np.ndarray,
        gravity: float,
        q_diag: np.ndarray,
        r_diag: np.ndarray,
        thrust_min: float,
        thrust_max: float,
        moment_min: np.ndarray,
        moment_max: np.ndarray,
        max_moment_rates: np.ndarray | None,
    ) -> None:
        self.mass = float(mass)
        self.inertia = np.asarray(inertia, dtype=float).reshape(-1)
        self.gravity = float(gravity)

        if self.inertia.shape != (3,):
            raise ValueError(f"inertia must have shape (3,), got {self.inertia.shape}.")

        self.Ix = float(self.inertia[0])
        self.Iy = float(self.inertia[1])
        self.Iz = float(self.inertia[2])

        if self.mass <= 0.0:
            raise ValueError(f"mass must be positive, got {self.mass}.")

        if self.Ix <= 0.0 or self.Iy <= 0.0 or self.Iz <= 0.0:
            raise ValueError(
                "All inertia components must be positive, "
                f"got inertia={self.inertia}."
            )

        self.hover_thrust = self.mass * self.gravity

        moment_min = np.asarray(moment_min, dtype=float).reshape(-1)
        moment_max = np.asarray(moment_max, dtype=float).reshape(-1)

        if moment_min.shape != (3,):
            raise ValueError(f"moment_min must have shape (3,), got {moment_min.shape}.")

        if moment_max.shape != (3,):
            raise ValueError(f"moment_max must have shape (3,), got {moment_max.shape}.")

        u_min = np.concatenate(
            [
                np.array([float(thrust_min)], dtype=float),
                moment_min,
            ]
        )
        u_max = np.concatenate(
            [
                np.array([float(thrust_max)], dtype=float),
                moment_max,
            ]
        )

        super().__init__(
            dt=dt,
            x_dim=12,
            u_dim=4,
            u_min=u_min,
            u_max=u_max,
            name="Quadrotor3DLQRHoverController",
        )

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
                f"thrust_min={thrust_min} is greater than "
                f"hover_thrust={self.hover_thrust}."
            )

        if thrust_max < self.hover_thrust:
            raise ValueError(
                f"thrust_max={thrust_max} is smaller than "
                f"hover_thrust={self.hover_thrust}."
            )

        if np.any(moment_min > 0.0):
            raise ValueError(f"moment_min should be <= 0 on all axes, got {moment_min}.")

        if np.any(moment_max < 0.0):
            raise ValueError(f"moment_max should be >= 0 on all axes, got {moment_max}.")

        if np.any(moment_min >= moment_max):
            raise ValueError(
                "Each component of moment_min must be smaller than moment_max, "
                f"got moment_min={moment_min}, moment_max={moment_max}."
            )

        if max_moment_rates is None:
            self.max_moment_rates = None
        else:
            self.max_moment_rates = np.asarray(max_moment_rates, dtype=float).reshape(-1)
            if self.max_moment_rates.shape != (3,):
                raise ValueError(
                    "max_moment_rates must have shape (3,), "
                    f"got {self.max_moment_rates.shape}."
                )
            if np.any(self.max_moment_rates <= 0.0):
                raise ValueError(
                    "max_moment_rates must be positive on all axes, "
                    f"got {self.max_moment_rates}."
                )

        self.prev_moments = np.zeros(3, dtype=float)

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
        self.prev_moments = np.zeros(3, dtype=float)

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
        self.prev_moments = np.zeros(3, dtype=float)

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

        u_eq = np.array(
            [
                self.hover_thrust,
                0.0,
                0.0,
                0.0,
            ],
            dtype=float,
        )

        u_unsat = u_eq - self.K @ error

        if self.max_moment_rates is not None:
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

        Ix = self.Ix
        Iy = self.Iy
        Iz = self.Iz

        A = np.zeros((self.x_dim, self.x_dim), dtype=float)
        B = np.zeros((self.x_dim, self.u_dim), dtype=float)

        # Position kinematics.
        A[0, 6] = 1.0   # px_dot = vx
        A[1, 7] = 1.0   # py_dot = vy
        A[2, 8] = 1.0   # pz_dot = vz

        # Attitude kinematics around hover.
        A[3, 9] = 1.0    # phi_dot = p
        A[4, 10] = 1.0   # theta_dot = q
        A[5, 11] = 1.0   # psi_dot = r

        # Translational linearized dynamics.
        A[6, 4] = g      # vx_dot ≈ g * theta
        A[7, 3] = -g     # vy_dot ≈ -g * phi

        # Thrust affects vertical acceleration.
        B[8, 0] = 1.0 / m

        # Moments affect angular accelerations.
        B[9, 1] = 1.0 / Ix
        B[10, 2] = 1.0 / Iy
        B[11, 3] = 1.0 / Iz

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

        moments = u_limited[1:4]
        dM = moments - self.prev_moments

        dM_max = self.max_moment_rates * self.dt
        dM = np.clip(dM, -dM_max, dM_max)

        moments_limited = self.prev_moments + dM
        u_limited[1:4] = moments_limited

        self.prev_moments = moments_limited.copy()

        return u_limited