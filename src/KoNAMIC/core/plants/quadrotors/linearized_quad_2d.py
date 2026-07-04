from __future__ import annotations

import numpy as np

from KoNAMIC.core.systems.drone.drone_spec import DroneSpec
from ..plant import Plant


class LinearizedQuad2D(Plant):
    """
    Linearized 2D planar quadrotor model around hover.

    Nonlinear state convention:
        x = [y, z, theta, y_dot, z_dot, theta_dot]

    Nonlinear input convention:
        u = [F, tau]

    This plant uses deviation coordinates:
        dx = x - x_eq
        du = u - u_eq

    Therefore, _dynamics(t, dx, du) returns:
        dx_dot = A dx + B du

    Hover equilibrium:
        x_eq = [y_eq, z_eq, 0, 0, 0, 0]
        u_eq = [m g, 0]

    Since the linearization is translation invariant in y and z, y_eq and z_eq
    are set to zero here.
    """

    def __init__(self, dt: float, system: DroneSpec):
        if system.sys_dim != 2:
            raise ValueError(
                f"LinearizedQuad2D expects a 2D drone, got drone_dim={system.sys_dim}"
            )

        super().__init__(
            dt=dt,
            system=system,
            discrete_or_continuous="continuous",
        )

    @property
    def mass(self) -> float:
        return self.system.mass

    @property
    def gravity(self) -> float:
        return self.system.gravity

    @property
    def inertia_zz(self) -> float:
        if self.system.inertia is None:
            raise ValueError("DroneSpec.inertia is required for LinearizedQuad2D")

        inertia = np.asarray(self.system.inertia, dtype=float)

        if inertia.shape == (3,):
            return float(inertia[2])

        if inertia.shape == (3, 3):
            return float(inertia[2, 2])

        raise ValueError(f"Unsupported inertia shape: {inertia.shape}")

    @property
    def equilibrium_state(self) -> np.ndarray:
        return np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=float,
        )

    @property
    def equilibrium_input(self) -> np.ndarray:
        return np.array(
            [self.mass * self.gravity, 0.0],
            dtype=float,
        )

    @property
    def A(self) -> np.ndarray:
        """
        Continuous-time linearized state matrix.

        State:
            dx = [dy, dz, dtheta, dy_dot, dz_dot, dtheta_dot]
        """

        A = np.zeros((6, 6), dtype=float)

        A[0, 3] = 1.0
        A[1, 4] = 1.0
        A[2, 5] = 1.0

        A[3, 2] = -self.gravity

        return A

    @property
    def B(self) -> np.ndarray:
        """
        Continuous-time linearized input matrix.

        Input:
            du = [dF, dtau]
        """

        B = np.zeros((6, 2), dtype=float)

        B[4, 0] = 1.0 / self.mass
        B[5, 1] = 1.0 / self.inertia_zz

        return B

    def to_deviation_state(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)

        if x.shape != (6,):
            raise ValueError(f"Expected state shape (6,), got {x.shape}")

        return x - self.equilibrium_state

    def to_absolute_state(self, dx: np.ndarray) -> np.ndarray:
        dx = np.asarray(dx, dtype=float)

        if dx.shape != (6,):
            raise ValueError(f"Expected deviation state shape (6,), got {dx.shape}")

        return self.equilibrium_state + dx

    def to_deviation_input(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)

        if u.shape != (2,):
            raise ValueError(f"Expected input shape (2,), got {u.shape}")

        return u - self.equilibrium_input

    def to_absolute_input(self, du: np.ndarray) -> np.ndarray:
        du = np.asarray(du, dtype=float)

        if du.shape != (2,):
            raise ValueError(f"Expected deviation input shape (2,), got {du.shape}")

        return self.equilibrium_input + du

    def _dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Linearized dynamics in deviation coordinates.

        Parameters
        ----------
        t:
            Time. Unused, kept for compatibility with Plant.
        x:
            Deviation state dx = x_abs - x_eq.
        u:
            Deviation input du = u_abs - u_eq.

        Returns
        -------
        dx_dot:
            Time derivative of the deviation state.
        """

        dx = np.asarray(x, dtype=float)
        du = np.asarray(u, dtype=float)

        if dx.shape != (6,):
            raise ValueError(f"Expected deviation state shape (6,), got {dx.shape}")

        if du.shape != (2,):
            raise ValueError(f"Expected deviation input shape (2,), got {du.shape}")

        return self.A @ dx + self.B @ du