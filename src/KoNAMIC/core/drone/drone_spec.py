from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Mapping
import numpy as np
import yaml

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
    name: str
    drone_dim: int
    mass: float
    gravity: float
    arm_length: float
    inertia: np.ndarray  # (3,) or (3,3)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DroneSpec":
        """
        Load a DroneSpec from a YAML configuration file.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Drone config file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ValueError(f"Drone config file is empty: {path}")

        if not isinstance(data, Mapping):
            raise TypeError(
                f"Expected YAML file to contain a mapping/dict, got {type(data).__name__}"
            )

        required_keys = {
            "name",
            "drone_dim",
            "mass",
            "gravity",
            "arm_length",
            "inertia",
        }

        missing_keys = required_keys - data.keys()
        if missing_keys:
            raise KeyError(
                f"Missing required key(s) in drone config {path}: "
                f"{sorted(missing_keys)}"
            )

        return cls(
            name=str(data["name"]),
            drone_dim=int(data["drone_dim"]),
            mass=float(data["mass"]),
            gravity=float(data["gravity"]),
            arm_length=float(data["arm_length"]),
            inertia=np.asarray(data["inertia"], dtype=float),
        )

    def __post_init__(self):
        if self.drone_dim not in (1, 2, 3):
            raise ValueError(f"Invalid drone_dim={self.drone_dim}")

        if self.mass <= 0:
            raise ValueError(f"mass must be > 0, got {self.mass}")

        if self.gravity <= 0:
            raise ValueError(f"gravity must be > 0, got {self.gravity}")

        if self.arm_length <= 0:
            raise ValueError(f"arm_length must be > 0, got {self.arm_length}")

        if self.inertia is not None:
            inertia = np.asarray(self.inertia)
            if inertia.shape not in [(3,), (3, 3)]:
                raise ValueError(
                    f"inertia must be (3,) or (3,3), got {inertia.shape}"
                )
            object.__setattr__(self, "inertia", inertia.astype(float))

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

    def get_x_labels(self, only_positions: bool = False) -> List[str]:
        return get_x_labels(self.drone_dim, only_positions)

    def get_u_labels(self) -> List[str]:
        return get_u_labels(self.drone_dim)

    @property
    def hover_thrust(self) -> float:
        return self.mass * self.gravity

    @property
    def angle_indexes(self) -> List[int]:
        return get_angle_indexes(self.drone_dim)

    def convert_state_to_deg(self, x: np.ndarray) -> np.ndarray:
        return convert_rad_to_deg_np(x, self.angle_indexes)

    def split_state(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        half = self.x_dim // 2
        return x[..., :half], x[..., half:]

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

    def convert_available_angles_to_deg(self, x: np.ndarray) -> np.ndarray:
        """
        Convert all available angle components from radians to degrees.

        This is useful for partial states or references that do not contain
        all state components.
        """
        angle_indexes = [
            idx for idx in self.angle_indexes
            if idx < x.shape[-1]
        ]
        return convert_rad_to_deg_np(x, angle_indexes)

    @property
    def inertia_matrix(self) -> Optional[np.ndarray]:
        if self.inertia is None:
            return None

        if self.inertia.shape == (3,):
            return np.diag(self.inertia)

        return self.inertia

    def __repr__(self) -> str:
        return (
            f"DroneSpec(name={self.name!r}, dim={self.drone_dim}, "
            f"x_dim={self.x_dim}, u_dim={self.u_dim})"
        )