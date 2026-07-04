from __future__ import annotations
import numpy as np

from KoNAMIC.core.systems.drone.drone_spec import DroneSpec
from ..plant import Plant


class Quad2D(Plant):
    """
    2D planar quad model.
    State: [y, z, theta, y_dot, z_dot, theta_dot] (6)
    Input: [F, tau] (2)
    """

    def __init__(self, dt: float, system: DroneSpec):
        if system.system_dim != 2:
            raise ValueError(
                f"PlanarQuad expects a 2D drone, got drone_dim={system.system_dim}"
            )
        super().__init__(dt=dt, system=system, discrete_or_continuous="continuous")

    @property
    def mass(self) -> float:
        return self.system.mass

    @property
    def inertia_zz(self) -> float:
        if self.system.inertia is None:
            raise ValueError("DroneSpec.inertia is required for PlanarQuad")

        inertia = np.asarray(self.system.inertia, dtype=float)
        if inertia.shape == (3,):
            return float(inertia[2])
        if inertia.shape == (3, 3):
            return float(inertia[2, 2])

        raise ValueError(f"Unsupported inertia shape: {inertia.shape}")

    @property
    def gravity(self) -> float:
        return self.system.gravity

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

