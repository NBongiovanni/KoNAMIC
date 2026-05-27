from __future__ import annotations

from typing import Any, Optional
import numpy as np

from .base_controller import BaseController


class PIDPlanarPosAttController(BaseController):
    """
    Contrôleur position + attitude pour planar quadrotor 2D.

    Etat supposé :
        x = [x, z, theta, x_dot, z_dot, theta_dot]

    Commande :
        u = [T, M]

    Référence supposée :
        x_ref = [x_ref, z_ref, theta_ref, x_dot_ref, z_dot_ref, theta_dot_ref]
    Seules les composantes x_ref et z_ref sont utilisées pour la boucle externe.
    theta_ref est généré par le contrôleur à partir de l'erreur en x.

    Convention dynamique supposée :
        x_ddot = -(T/m) * sin(theta)
        z_ddot =  (T/m) * cos(theta) - g

    Pour petits angles :
        x_ddot ≈ -(T/m) * theta
    donc :
        theta_cmd ≈ - a_x_cmd / (T/m)
    """

    def __init__(
        self,
        dt: float,
        mass: float,
        inertia_y: float,
        gravity: float,
        kp_pos: np.ndarray,      # [Kp_x, Kp_z]
        ki_pos: np.ndarray,      # [Ki_x, Ki_z]
        kd_pos: np.ndarray,      # [Kd_x, Kd_z]
        kp_att: float,           # Kp_theta
        ki_att: float,           # Ki_theta, mettre 0.0 pour PD
        kd_att: float,           # Kd_theta
        deriv_filter_n: float,
        theta_max: float,
        thrust_min: float,
        thrust_max: float,
        moment_max: float,
        acc_x_max: float,
        att_cmd_alpha: float = 1.0,
        max_moment_rate: Optional[float] = None,
        x_dim: int = 6,
        u_dim: int = 2,
    ) -> None:

        super().__init__(
            dt=dt,
            x_dim=x_dim,
            u_dim=u_dim,
            u_min=None,
            u_max=None,
            name="PIDPlanarPosAttController",
        )

        if x_dim != 6:
            raise ValueError("PIDPlanarPosAttController expects x_dim=6.")
        if u_dim != 2:
            raise ValueError("PIDPlanarPosAttController expects u_dim=2.")

        self.m = float(mass)
        self.Iy = float(inertia_y)
        self.g = float(gravity)
        self.hover_thrust = self.m * self.g

        self.kp_pos = np.asarray(kp_pos, dtype=float).reshape(2)
        self.ki_pos = np.asarray(ki_pos, dtype=float).reshape(2)
        self.kd_pos = np.asarray(kd_pos, dtype=float).reshape(2)

        self.kp_att = float(kp_att)
        self.ki_att = float(ki_att)
        self.kd_att = float(kd_att)

        self.N = float(deriv_filter_n)

        self.theta_ref_max = float(theta_max)
        self.thrust_min = float(thrust_min)
        self.thrust_max = float(thrust_max)

        self.moment_max = float(moment_max)
        self.moment_min = -float(moment_max)

        self.acc_x_max = float(acc_x_max)
        self.att_cmd_alpha = float(att_cmd_alpha)

        self.max_moment_rate = None if max_moment_rate is None else float(max_moment_rate)

        self.x_ref_traj: Optional[np.ndarray] = None
        self.step_idx: int = 0

        self.int_pos = np.zeros(2, dtype=float)   # [int_x, int_z]
        self.int_att = 0.0

        self.d_pos_state = np.zeros(2, dtype=float)
        self.d_att_state = 0.0

        self.theta_c_f = 0.0
        self.prev_moment = 0.0

        self.x_traj: list[np.ndarray] = []
        self.err_pos_traj: list[np.ndarray] = []
        self.err_att_traj: list[float] = []

    def reset(self) -> None:
        super().reset()

        self.x_ref_traj = None
        self.step_idx = 0

        self.int_pos[:] = 0.0
        self.int_att = 0.0

        self.d_pos_state[:] = 0.0
        self.d_att_state = 0.0

        self.theta_c_f = 0.0
        self.prev_moment = 0.0

        self.x_traj.clear()
        self.err_pos_traj.clear()
        self.err_att_traj.clear()

    def build(self) -> None:
        self.is_built = True

    def set_reference(self, x_ref_traj: np.ndarray) -> None:
        x_ref_traj = np.asarray(x_ref_traj, dtype=float)

        if x_ref_traj.ndim != 2:
            raise ValueError("x_ref_traj must be 2D, shape (T, x_dim).")
        if x_ref_traj.shape[1] != self.x_dim:
            raise ValueError(f"x_ref_traj must have shape (T, {self.x_dim}).")
        if x_ref_traj.shape[0] == 0:
            raise ValueError("x_ref_traj must contain at least one time step.")

        self.x_ref_traj = x_ref_traj.copy()
        self.step_idx = 0
        self.prev_moment = 0.0

        super().set_reference(self.x_ref_traj)

    def set_initial_conditions(self, observation: Any) -> None:
        x0 = self._extract_state_from_observation(observation)
        super().set_initial_conditions(x0)

        self.step_idx = 0

        self.int_pos[:] = 0.0
        self.int_att = 0.0

        self.d_pos_state[:] = 0.0
        self.d_att_state = 0.0

        self.theta_c_f = float(x0[2])
        self.prev_moment = 0.0

    def compute_control(self, observation: Any) -> np.ndarray:
        x_k = self._extract_state_from_observation(observation)

        if x_k.shape != (self.x_dim,):
            raise ValueError(f"x_k must have shape ({self.x_dim},), got {x_k.shape}.")

        if self.x_ref_traj is None:
            raise RuntimeError("Reference trajectory not set. Call set_reference(...) first.")

        k = min(self.step_idx, self.x_ref_traj.shape[0] - 1)
        x_ref_k = self.x_ref_traj[k]

        x, z, theta, x_dot, z_dot, theta_dot = x_k

        x_ref = x_ref_k[0]
        z_ref = x_ref_k[1]

        # -----------------------------
        # Boucle externe position
        # -----------------------------
        e_pos = np.array([x_ref - x, z_ref - z], dtype=float)

        self.int_pos += self.dt * e_pos

        d_pos = np.zeros(2, dtype=float)
        for i in range(2):
            d_pos[i], self.d_pos_state[i] = self._filtered_derivative(
                e=e_pos[i],
                x_d=self.d_pos_state[i],
            )

        # Accélération horizontale désirée
        a_x_cmd = (
            self.kp_pos[0] * e_pos[0]
            + self.ki_pos[0] * self.int_pos[0]
            + self.kd_pos[0] * d_pos[0]
        )
        a_x_cmd = float(np.clip(a_x_cmd, -self.acc_x_max, self.acc_x_max))

        # Poussée verticale autour du hover
        T_unsat = (
            self.kp_pos[1] * e_pos[1]
            + self.ki_pos[1] * self.int_pos[1]
            + self.kd_pos[1] * d_pos[1]
            + self.hover_thrust
        )
        T = float(np.clip(T_unsat, self.thrust_min, self.thrust_max))

        # -----------------------------
        # Conversion x -> consigne theta
        # -----------------------------
        # Avec la convention x_ddot = -(T/m) sin(theta),
        # pour petits angles : theta_cmd = -a_x_cmd / (T/m)
        specific_thrust = max(T / self.m, 1e-6)

        theta_c = -a_x_cmd / specific_thrust
        theta_c = float(np.clip(theta_c, -self.theta_ref_max, self.theta_ref_max))

        alpha = self.att_cmd_alpha
        self.theta_c_f = (1.0 - alpha) * self.theta_c_f + alpha * theta_c

        # -----------------------------
        # Boucle interne attitude
        # -----------------------------
        e_theta = self.theta_c_f - theta

        self.int_att += self.dt * e_theta

        d_theta, self.d_att_state = self._filtered_derivative(
            e=e_theta,
            x_d=self.d_att_state,
        )

        theta_ddot_cmd = (
            self.kp_att * e_theta
            + self.ki_att * self.int_att
            + self.kd_att * d_theta
        )

        M_unsat = self.Iy * theta_ddot_cmd
        M = float(np.clip(M_unsat, self.moment_min, self.moment_max))

        # Limitation optionnelle de la vitesse de variation du moment
        if self.max_moment_rate is not None:
            dM = M - self.prev_moment
            dM_max = self.max_moment_rate * self.dt
            dM = float(np.clip(dM, -dM_max, dM_max))
            M = self.prev_moment + dM

        self.prev_moment = M

        u = np.array([T, M], dtype=float)

        self.step_idx += 1

        self.x_traj.append(x_k.copy())
        self.err_pos_traj.append(e_pos.copy())
        self.err_att_traj.append(float(e_theta))

        # Optionnel : stocker la consigne theta dans la trajectoire de référence
        # si votre pipeline visualise x_ref_traj[:, 2].
        self.x_ref_traj[k, 2] = self.theta_c_f

        return self._store_control(
            u,
            info={
                "x_ref_k": x_ref_k.copy(),
                "e_pos": e_pos.copy(),
                "e_theta": float(e_theta),
                "theta_c": float(theta_c),
                "theta_c_f": float(self.theta_c_f),
                "T_unsat": float(T_unsat),
                "M_unsat": float(M_unsat),
                "state": np.array([x, z, theta, x_dot, z_dot, theta_dot], dtype=float),
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

    def _filtered_derivative(self, e: float, x_d: float) -> tuple[float, float]:
        x_d_new = x_d + self.dt * self.N * (e - x_d)
        e_dot_f = self.N * (e - x_d)
        return float(e_dot_f), float(x_d_new)