from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Mapping
import numpy as np
import yaml

from ..system_spec import SystemSpec
from .dimensions import get_sys_dimensions, get_num_views
from .labels import get_x_labels, get_u_labels, get_x_names
from .state_layout import get_angle_indexes, convert_rad_to_deg_np


@dataclass(frozen=True)
class DroneSpec(SystemSpec):
    system_name: str
    system_type: str
    system_dim: int
    mass: float
    gravity: float
    arm_length: float
    inertia: np.ndarray
    state_dim: int
    input_dim: int
    state_names: list[str]
    input_names: list[str]
    ref_names: list[str]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DroneSpec":
        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        return cls.from_dict(cfg)

    @classmethod
    def from_dict(cls, cfg: dict) -> "DroneSpec":
        if cfg.get("system_type") != "quadrotor":
            raise ValueError(
                f"DroneSpec expects system_type='quadrotor', "
                f"got {cfg.get('system_type')!r}."
            )

        return cls(**cfg)

    def __post_init__(self):
        if self.system_dim not in (1, 2, 3):
            raise ValueError(f"Invalid sys_dim={self.system_dim}")

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
        x_dim, _, _ = get_sys_dimensions(self.system_dim)
        return x_dim

    @property
    def u_dim(self) -> int:
        _, u_dim, _ = get_sys_dimensions(self.system_dim)
        return u_dim

    @property
    def x_ref_dim_closed_loop(self) -> int:
        _, _, x_ref_dim = get_sys_dimensions(self.system_dim, task="control")
        return x_ref_dim

    @property
    def x_ref_dim_open_loop(self) -> int:
        _, _, x_ref_dim = get_sys_dimensions(self.system_dim, task="open_loop")
        return x_ref_dim

    @property
    def num_views(self) -> int:
        return get_num_views(self.system_dim)

    def get_x_labels(self, only_positions: bool = False) -> List[str]:
        return get_x_labels(self.system_dim, only_positions)

    def get_x_names(self, only_positions: bool = False) -> List[str]:
        return get_x_names(self.system_dim, only_positions)

    def get_u_labels(self) -> List[str]:
        return get_u_labels(self.system_dim)

    @property
    def hover_thrust(self) -> float:
        return self.mass * self.gravity

    @property
    def angle_indexes(self) -> List[int]:
        return get_angle_indexes(self.system_dim)

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
            f"DroneSpec(name={self.system_name!r}, dim={self.system_dim}, "
            f"x_dim={self.x_dim}, u_dim={self.u_dim})"
        )

    def get_input_plot_groups(self, group_inputs: bool = True) -> list[dict]:
        labels = self.get_u_labels()

        if not group_inputs:
            return [
                {
                    "indices": [i],
                    "label": labels[i],
                }
                for i in range(self.u_dim)
            ]

        if self.system_dim == 2:
            return [
                {
                    "indices": [0],
                    "label": labels[0],
                },
                {
                    "indices": [1],
                    "label": labels[1],
                },
            ]

        if self.system_dim == 3:
            return [
                {
                    "indices": [0],
                    "label": labels[0],
                },
                {
                    "indices": [1, 2, 3],
                    "label": "Moments",
                },
            ]

        return super().get_input_plot_groups(group_inputs=group_inputs)