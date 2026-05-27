from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from .dimensions import get_dimensions, get_num_views
from .labels import get_x_labels, get_u_labels
from .state_layout import get_angle_indexes, convert_rad_to_deg_np


@dataclass(frozen=True)
class DroneSpec:
    """
    Minimal and robust specification of a drone.

    This class is the single source of truth for:
    - physical parameters
    - state/control dimensions
    - labels and state structure

    It is intentionally minimal: only parameters that are
    actually used in the project are included.
    """

    # ------------------------------------------------------------------
    # Core definition
    # ------------------------------------------------------------------
    name: str
    drone_dim: int

    # ------------------------------------------------------------------
    # Physical parameters
    # ------------------------------------------------------------------
    mass: float
    gravity: float
    arm_length: float
    inertia: np.ndarray  # (3,) or (3,3)

    # Control limits (optional but useful)
    u_min: Optional[np.ndarray] = None
    u_max: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def __post_init__(self):
        if self.drone_dim not in (1, 2, 3):
            raise ValueError(f"Invalid drone_dim={self.drone_dim}")

        if self.mass <= 0:
            raise ValueError(f"mass must be > 0, got {self.mass}")

        if self.gravity <= 0:
            raise ValueError(f"gravity must be > 0, got {self.gravity}")

        if self.inertia is not None:
            inertia = np.asarray(self.inertia)
            if inertia.shape not in [(3,), (3, 3)]:
                raise ValueError(
                    f"inertia must be (3,) or (3,3), got {inertia.shape}"
                )
            object.__setattr__(self, "inertia", inertia.astype(float))

        if self.u_min is not None:
            u_min = np.asarray(self.u_min, dtype=float)
            if u_min.shape != (self.u_dim,):
                raise ValueError(
                    f"u_min must have shape ({self.u_dim},), got {u_min.shape}"
                )
            object.__setattr__(self, "u_min", u_min)

        if self.u_max is not None:
            u_max = np.asarray(self.u_max, dtype=float)
            if u_max.shape != (self.u_dim,):
                raise ValueError(
                    f"u_max must have shape ({self.u_dim},), got {u_max.shape}"
                )
            object.__setattr__(self, "u_max", u_max)

        if self.u_min is not None and self.u_max is not None:
            if np.any(self.u_min > self.u_max):
                raise ValueError("u_min must be <= u_max")

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------
    @property
    def x_dim(self) -> int:
        x_dim, _, _ = get_dimensions(self.drone_dim)
        return x_dim

    @property
    def u_dim(self) -> int:
        _, u_dim, _ = get_dimensions(self.drone_dim)
        return u_dim

    @property
    def x_ref_dim_closed_loop(self) -> int:
        _, _, x_ref_dim = get_dimensions(self.drone_dim, task="control")
        return x_ref_dim

    @property
    def x_ref_dim_open_loop(self) -> int:
        _, _, x_ref_dim = get_dimensions(self.drone_dim, task="open_loop")
        return x_ref_dim

    @property
    def num_views(self) -> int:
        return get_num_views(self.drone_dim)

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------
    def get_x_labels(self, only_positions: bool = False) -> List[str]:
        return get_x_labels(self.drone_dim, only_positions)

    def get_u_labels(self) -> List[str]:
        return get_u_labels(self.drone_dim)

    @property
    def hover_thrust(self) -> float:
        return self.mass * self.gravity

    # ------------------------------------------------------------------
    # State structure
    # ------------------------------------------------------------------
    @property
    def angle_indexes(self) -> List[int]:
        return get_angle_indexes(self.drone_dim)

    def convert_state_to_deg(self, x: np.ndarray) -> np.ndarray:
        return convert_rad_to_deg_np(x, self.angle_indexes)

    def split_state(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split state into (positions/orientation, velocities)
        """
        half = self.x_dim // 2
        return x[..., :half], x[..., half:]

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def check_state_dim(self, x: np.ndarray):
        if x.shape[-1] != self.x_dim:
            raise ValueError(
                f"Expected state dim {self.x_dim}, got {x.shape[-1]}"
            )

    def check_control_dim(self, u: np.ndarray):
        if u.shape[-1] != self.u_dim:
            raise ValueError(
                f"Expected control dim {self.u_dim}, got {u.shape[-1]}"
            )

    # ------------------------------------------------------------------
    # Physics helpers (minimal but useful)
    # ------------------------------------------------------------------
    @property
    def inertia_matrix(self) -> Optional[np.ndarray]:
        """
        Return inertia as a full matrix.
        """
        if self.inertia is None:
            return None

        if self.inertia.shape == (3,):
            return np.diag(self.inertia)

        return self.inertia

    # ------------------------------------------------------------------
    # Debug / display
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"DroneSpec(name={self.name!r}, dim={self.drone_dim}, "
            f"x_dim={self.x_dim}, u_dim={self.u_dim})"
        )