from __future__ import annotations

from typing import Any, Optional
import numpy as np

from .base_controller import BaseController


class PIDPosAttController(BaseController):
    """
    Contrôleur position + attitude pour drone.

    Etat supposé :
        x = [x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot]

    Commande :
        u = [Tz, Mx, My, Mz]

    Référence supposée :
        x_ref = [x_ref, y_ref, z_ref, psi_ref, ...]
    Seules les composantes x, y, z, psi sont utilisées ici.
    """

    def __init__(
        self,
        dt: float,
        x_dim: int,
        u_dim: int,
        mass: float,
        inertia: np.ndarray,
        gravity: float,
        kp_pos: np.ndarray,   # [Kp_x, Kp_y, Kp_z]
        ki_pos: np.ndarray,   # [Ki_x, Ki_y, Ki_z]
        kd_pos: np.ndarray,   # [Kd_x, Kd_y, Kd_z]
        kp_att: np.ndarray,   # [Kp_phi, Kp_theta, Kp_psi]
        ki_att: np.ndarray,   # [Ki_phi, Ki_theta, Ki_psi]
        kd_att: np.ndarray,   # [Kd_phi, Kd_theta, Kd_psi]
        deriv_filter_n: float,
        phi_max: float,
        theta_max: float,
        thrust_min: float,
        thrust_max: float,
        att_cmd_alpha: float,
        moment_max: np.ndarray,   # [Mx_max, My_max, Mz_max]
        acc_xy_max: float,
        max_moment_rate: np.ndarray,
        yaw_wrap: bool = True,
    ) -> None:

        super().__init__(
            dt=dt,
            x_dim=x_dim,
            u_dim=u_dim,
            u_min=None,
            u_max=None,
            name="PIDPosAttController",
        )

        self.m = float(mass)
        self.g = float(gravity)
        self.inertia = inertia
        self.hover_thrust = self.m * self.g

        self.kp_pos = kp_pos
        self.ki_pos = ki_pos
        self.kd_pos = kd_pos

        self.kp_att = kp_att
        self.ki_att = ki_att
        self.kd_att = kd_att

        self.N = float(deriv_filter_n)
        self.phi_ref_max = float(phi_max)
        self.theta_ref_max = float(theta_max)
        self.thrust_min = float(thrust_min)
        self.thrust_max = float(thrust_max)
        self.acc_xy_max = float(acc_xy_max)
        self.yaw_wrap = bool(yaw_wrap)

        self.moment_max = moment_max
        self.moment_min = (-1)*moment_max


        self.x_ref_traj: Optional[np.ndarray] = None
        self.step_idx: int = 0

        self.int_pos = np.zeros(3, dtype=float)
        self.int_att = np.zeros(3, dtype=float)

        # états internes du filtre dérivatif façon Simulink
        self.d_pos_state = np.zeros(3, dtype=float)   # x_d_x, x_d_y, x_d_z
        self.d_att_state = np.zeros(3, dtype=float)   # x_d_phi, x_d_theta, x_d_psi

        self.x_traj: list[np.ndarray] = []
        self.u_traj: list[np.ndarray] = []
        self.err_pos_traj: list[np.ndarray] = []
        self.err_att_traj: list[np.ndarray] = []

        self.phi_c_f = 0.0
        self.theta_c_f = 0.0
        self.att_cmd_alpha = att_cmd_alpha

        self.prev_acc_cmd = np.zeros(3, dtype=float)

        # limites de vitesse de variation des moments [N.m/s]
        self.max_moment_rate = max_moment_rate

    def reset(self) -> None:
        super().reset()
        self.x_ref_traj = None
        self.step_idx = 0
        self.int_pos[:] = 0.0
        self.int_att[:] = 0.0
        self.d_pos_state[:] = 0.0
        self.d_att_state[:] = 0.0
        self.x_traj.clear()
        self.u_traj.clear()
        self.err_pos_traj.clear()
        self.err_att_traj.clear()
        self.phi_c_f = 0.0
        self.theta_c_f = 0.0

    def build(self) -> None:
        self.is_built = True

    def set_reference(self, x_ref_traj: np.ndarray) -> None:
        x_ref_traj = np.asarray(x_ref_traj, dtype=float)
        if x_ref_traj.ndim != 2:
            raise ValueError("x_ref_traj must be 2D, shape (T, x_dim)")
        if x_ref_traj.shape[1] != self.x_dim:
            raise ValueError(f"x_ref_traj must have shape (T, {self.x_dim})")
        if x_ref_traj.shape[0] == 0:
            raise ValueError("x_ref_traj must contain at least one time step")

        self.x_ref_traj = x_ref_traj
        self.step_idx = 0
        super().set_reference(x_ref_traj)
        self.prev_acc_cmd[:] = 0.0

    def set_initial_conditions(self, observation: Any) -> None:
        x0 = self._extract_state_from_observation(observation)
        super().set_initial_conditions(x0)
        self.step_idx = 0
        self.int_pos[:] = 0.0
        self.int_att[:] = 0.0
        self.d_pos_state[:] = 0.0
        self.d_att_state[:] = 0.0
        self.phi_c_f = float(x0[3])
        self.theta_c_f = float(x0[4])
        self.prev_acc_cmd[:] = 0.0

    def compute_control(self, observation: Any) -> np.ndarray:
        x_k = self._extract_state_from_observation(observation)

        if x_k.shape != (self.x_dim,):
            raise ValueError(f"x_k must have shape ({self.x_dim},), got {x_k.shape}")

        if self.x_ref_traj is None:
            raise RuntimeError("Reference trajectory not set. Call set_reference(...) first.")

        k = min(self.step_idx, self.x_ref_traj.shape[0] - 1)
        x_ref_k = self.x_ref_traj[k]

        alpha = self.att_cmd_alpha
        x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, p, q, r = x_k

        # Référence : on utilise x, y, z, psi
        x_ref = x_ref_k[0]
        y_ref = x_ref_k[1]
        z_ref = x_ref_k[2]
        psi_ref = 0.0

        # -----------------------------
        # Boucle position
        # -----------------------------
        e_pos = np.array([x_ref - x, y_ref - y, z_ref - z], dtype=float)

        self.int_pos += self.dt * e_pos

        d_pos = np.zeros(3, dtype=float)
        for i in range(3):
            d_pos[i], self.d_pos_state[i] = self._filtered_derivative(
                e=e_pos[i],
                x_d=self.d_pos_state[i],
            )

        a_xy_cmd = (
                self.kp_pos[:2] * e_pos[:2]
                + self.ki_pos[:2] * self.int_pos[:2]
                + self.kd_pos[:2] * d_pos[:2]
        )
        a_xy_cmd = np.clip(a_xy_cmd, -self.acc_xy_max, self.acc_xy_max)

        Tz_unsat = (
                self.kp_pos[2] * e_pos[2]
                + self.ki_pos[2] * self.int_pos[2]
                + self.kd_pos[2] * d_pos[2]
                + self.hover_thrust
        )
        Tz = float(np.clip(Tz_unsat, self.thrust_min, self.thrust_max))

        # ---------------------------------------
        # Conversion XY -> consignes attitude
        # ---------------------------------------
        cpsi = np.cos(psi)
        spsi = np.sin(psi)

        ax_body_yaw = cpsi * a_xy_cmd[0] + spsi * a_xy_cmd[1]
        ay_body_yaw = -spsi * a_xy_cmd[0] + cpsi * a_xy_cmd[1]

        U1 = Tz / self.m

        theta_c = np.clip(ax_body_yaw / U1, -self.theta_ref_max, self.theta_ref_max)
        phi_c = np.clip(-ay_body_yaw / U1, -self.phi_ref_max, self.phi_ref_max)

        self.theta_c_f = (1.0 - alpha) * self.theta_c_f + alpha * theta_c
        self.phi_c_f = (1.0 - alpha) * self.phi_c_f + alpha * phi_c

        # -----------------------------
        # Boucle attitude
        # Damping sur p, q, r
        # -----------------------------
        e_phi = self.phi_c_f - phi
        e_theta = self.theta_c_f - theta
        e_psi = self._angle_error(psi_ref, psi) if self.yaw_wrap else (psi_ref - psi)

        e_att = np.array([e_phi, e_theta, e_psi], dtype=float)

        d_att = np.zeros(3, dtype=float)
        for i in range(3):
            d_att[i], self.d_att_state[i] = self._filtered_derivative(
                e=e_att[i],
                x_d=self.d_att_state[i],
            )

        # Logique MATLAB : P + D sur dérivée d'erreur filtrée, sans intégrale
        acc_x = self.kp_att[0] * e_att[0] + self.kd_att[0] * d_att[0]
        acc_y = self.kp_att[1] * e_att[1] + self.kd_att[1] * d_att[1]
        acc_z = self.kp_att[2] * e_att[2] + self.kd_att[2] * d_att[2]

        acc_cmd = np.array([acc_x, acc_y, acc_z], dtype=float)
        acc_cmd = np.clip(acc_cmd, self.moment_min/self.inertia[0], self.moment_max/self.inertia[0])

        # limitation de vitesse de variation
        dacc_cmd = acc_cmd - self.prev_acc_cmd
        dacc_cmd_max = self.max_moment_rate * self.dt
        dacc_cmd = np.clip(dacc_cmd, -dacc_cmd_max/self.inertia[0], dacc_cmd_max/self.inertia[0])

        acc_cmd = self.prev_acc_cmd + dacc_cmd
        self.prev_acc_cmd = acc_cmd.copy()

        I_x, I_y, I_z = self.inertia
        u = np.array([Tz, acc_cmd[0]*I_x, acc_cmd[1]*I_y, acc_cmd[2]*I_z], dtype=float)

        self.step_idx += 1
        self.x_traj.append(x_k.copy())
        self.err_pos_traj.append(e_pos.copy())
        self.err_att_traj.append(e_att.copy())

        self.x_ref_traj[k][3] = self.phi_c_f
        self.x_ref_traj[k][4] = self.theta_c_f

        return self._store_control(
            u,
            info={
                "x_ref_k": x_ref_k.copy(),
                "e_pos": e_pos.copy(),
                "e_att": e_att.copy(),
                "phi_c": float(phi_c),
                "theta_c": float(theta_c),
                "Tz_unsat": float(Tz_unsat),
                "body_rates": np.array([p, q, r], dtype=float),
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
        # même idée que dans ton MATLAB :
        # x_d_new = x_d + Ts * N * (e - x_d)
        # e_dot_f = N * (e - x_d)
        x_d_new = x_d + self.dt * self.N * (e - x_d)
        e_dot_f = self.N * (e - x_d)
        return float(e_dot_f), float(x_d_new)

    @staticmethod
    def _angle_error(angle_ref: float, angle: float) -> float:
        return float(np.arctan2(np.sin(angle_ref - angle), np.cos(angle_ref - angle)))