from __future__ import annotations

import numpy as np

from KoNAMIC.core.systems.cartpole.cartpole_spec import CartPoleSpec
from .plant import Plant


class CartPole(Plant):
    """
    Standard cartpole dynamics with theta = 0 corresponding to the upright position.

    State:
        x = [p, theta, p_dot, theta_dot]

    Input:
        u = [force]

    Convention:
        theta = 0 is the unstable upright equilibrium.
        Positive force pushes the cart in the positive p direction.
    """

    def __init__(self, dt: float, system: CartPoleSpec):
        if system.x_dim != 4:
            raise ValueError(
                f"CartPole expects x_dim=4, got x_dim={system.x_dim}"
            )
        if system.u_dim != 1:
            raise ValueError(
                f"CartPole expects u_dim=1, got u_dim={system.u_dim}"
            )

        super().__init__(
            dt=dt,
            system=system,
            discrete_or_continuous="continuous",
        )

    @property
    def cartpole(self) -> CartPoleSpec:
        return self.system

    @property
    def cart_mass(self) -> float:
        return self.cartpole.cart_mass

    @property
    def pole_mass(self) -> float:
        return self.cartpole.pole_mass

    @property
    def pole_length(self) -> float:
        return self.cartpole.pole_length

    @property
    def gravity(self) -> float:
        return self.cartpole.gravity

    def _dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        p, theta, p_dot, theta_dot = x
        force = float(u[0])

        M = self.cart_mass
        m = self.pole_mass
        l = self.pole_length
        g = self.gravity

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        denominator = M + m * sin_theta**2

        p_ddot = (
            force
            - m * sin_theta * (l * theta_dot**2 + g * cos_theta)
        ) / denominator

        theta_ddot = (
            force * cos_theta
            - m * l * theta_dot**2 * sin_theta * cos_theta
            + (M + m) * g * sin_theta
        ) / (l * denominator)

        return np.array(
            [p_dot, theta_dot, p_ddot, theta_ddot],
            dtype=float,
        )