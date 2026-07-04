from __future__ import annotations

import numpy as np


class Plant:
    """
    Base class for plants (continuous or discrete).
    Enforces a single dynamics signature: f(t, x, u) -> x_dot or x_next.
    """

    def __init__(
        self,
        dt: float,
        system,
        discrete_or_continuous: str = "continuous",
    ):
        self.dt = float(dt)
        self.discrete_or_continuous = discrete_or_continuous
        self.system = system

        # Derived from DroneSpec when available
        self.system_dim: int | None = system.system_dim if system is not None else None
        self.x_dim: int | None = system.x_dim if system is not None else None
        self.u_dim: int | None = system.u_dim if system is not None else None

    def _dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        If continuous: return x_dot.
        If discrete: return x_next.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def update_state(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        u = np.asarray(u, dtype=float).reshape(-1)

        if self.x_dim is not None and x.shape[0] != self.x_dim:
            raise ValueError(f"x has dim {x.shape[0]} but expected {self.x_dim}")
        if self.u_dim is not None and u.shape[0] != self.u_dim:
            raise ValueError(f"u has dim {u.shape[0]} but expected {self.u_dim}")

        if self.discrete_or_continuous == "continuous":
            dt_small = self.dt
            x_next = x.copy()
            for _ in range(1):
                dx = self._dynamics(0.0, x_next, u)
                x_next = x_next + dt_small * dx
            return x_next

        if self.discrete_or_continuous == "discrete":
            return np.asarray(self._dynamics(0.0, x, u), dtype=float).reshape(-1)

        raise ValueError(f"Unknown discrete_or_continuous={self.discrete_or_continuous}")