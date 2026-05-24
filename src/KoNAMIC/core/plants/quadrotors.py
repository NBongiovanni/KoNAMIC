from __future__ import annotations
import numpy as np

from KoNAMIC.core.drone import DroneSpec
from .plant import Plant


class PlanarQuad(Plant):
    """
    2D planar quad model.
    State: [y, z, theta, y_dot, z_dot, theta_dot] (6)
    Input: [F, tau] (2)
    """

    def __init__(self, dt: float, drone: DroneSpec):
        if drone.drone_dim != 2:
            raise ValueError(
                f"PlanarQuad expects a 2D drone, got drone_dim={drone.drone_dim}"
            )
        super().__init__(dt=dt, drone=drone, discrete_or_continuous="continuous")

    @property
    def mass(self) -> float:
        return self.drone.mass

    @property
    def inertia_zz(self) -> float:
        if self.drone.inertia is None:
            raise ValueError("DroneSpec.inertia is required for PlanarQuad")

        inertia = np.asarray(self.drone.inertia, dtype=float)
        if inertia.shape == (3,):
            return float(inertia[2])
        if inertia.shape == (3, 3):
            return float(inertia[2, 2])

        raise ValueError(f"Unsupported inertia shape: {inertia.shape}")

    @property
    def gravity(self) -> float:
        return self.drone.gravity

    def _dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        y, z, theta, y_dot, z_dot, theta_dot = x
        F, tau = u

        y_ddot = -(F / self.mass) * np.sin(theta)
        z_ddot = (F / self.mass) * np.cos(theta) - self.gravity
        theta_ddot = tau / self.inertia_zz

        return np.array(
            [y_dot, z_dot, theta_dot, y_ddot, z_ddot, theta_ddot],
            dtype=float,
        )


class Quad3D(Plant):
    """
    Simple 3D quadrotor model.
    State (12): [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]
    Input (4): [F, tau_x, tau_y, tau_z]
    """

    def __init__(self, dt: float, drone: DroneSpec):
        if drone.drone_dim != 3:
            raise ValueError(
                f"Quad3D expects a 3D drone, got drone_dim={drone.drone_dim}"
            )
        super().__init__(dt=dt, drone=drone, discrete_or_continuous="continuous")

    @property
    def mass(self) -> float:
        return self.drone.mass

    @property
    def gravity(self) -> float:
        return self.drone.gravity

    @property
    def inertia_diag(self) -> tuple[float, float, float]:
        if self.drone.inertia is None:
            raise ValueError("DroneSpec.inertia is required for Quad3D")

        inertia = np.asarray(self.drone.inertia, dtype=float)
        if inertia.shape == (3,):
            return float(inertia[0]), float(inertia[1]), float(inertia[2])
        if inertia.shape == (3, 3):
            return float(inertia[0, 0]), float(inertia[1, 1]), float(inertia[2, 2])

        raise ValueError(f"Unsupported inertia shape: {inertia.shape}")

    @staticmethod
    def _rot_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
        cph, sph = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta), np.sin(theta)
        cps, sps = np.cos(psi), np.sin(psi)

        Rz = np.array([[cps, -sps, 0.0], [sps, cps, 0.0], [0.0, 0.0, 1.0]])
        Ry = np.array([[cth, 0.0, sth], [0.0, 1.0, 0.0], [-sth, 0.0, cth]])
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cph, -sph], [0.0, sph, cph]])
        return Rz @ Ry @ Rx

    @staticmethod
    def _euler_rates_matrix(phi: float, theta: float) -> np.ndarray:
        cth = np.cos(theta)
        if abs(cth) < 1e-6:
            cth = np.sign(cth) * 1e-6 if cth != 0 else 1e-6

        tth = np.tan(theta)
        cph, sph = np.cos(phi), np.sin(phi)

        return np.array(
            [
                [1.0, sph * tth, cph * tth],
                [0.0, cph, -sph],
                [0.0, sph / cth, cph / cth],
            ],
            dtype=float,
        )

    def _dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        px, py, pz, phi, theta, psi, vx, vy, vz, p, q, r = x
        F, tau_x, tau_y, tau_z = u
        Ix, Iy, Iz = self.inertia_diag

        R = self._rot_matrix(phi, theta, psi)
        thrust_world = R @ np.array([0.0, 0.0, F], dtype=float)

        ax, ay, az = thrust_world / self.mass - np.array([0.0, 0.0, self.gravity], dtype=float)

        T = self._euler_rates_matrix(phi, theta)
        euler_dot = T @ np.array([p, q, r], dtype=float)
        phi_dot, theta_dot, psi_dot = euler_dot

        omega = np.array([p, q, r], dtype=float)
        tau = np.array([tau_x, tau_y, tau_z], dtype=float)

        I = np.diag([Ix, Iy, Iz])
        omega_dot = np.linalg.solve(I, tau - np.cross(omega, I @ omega))
        # ω̇ = I⁻¹(τ − ω × (Iω)), solved via np.linalg.solve for better numerical stability than explicit inversion
        p_dot, q_dot, r_dot = omega_dot

        return np.array(
            [
                vx, vy, vz,
                phi_dot, theta_dot, psi_dot,
                ax, ay, az,
                p_dot, q_dot, r_dot,
            ],
            dtype=float,
        )