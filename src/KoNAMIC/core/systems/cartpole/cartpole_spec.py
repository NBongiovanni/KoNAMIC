from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Mapping
import numpy as np
import yaml

from .dimensions import get_cartpole_dimensions, get_num_views
from .labels import get_x_labels, get_u_labels, get_x_names
from .state_layout import get_angle_indexes, convert_rad_to_deg_np
from ..system_spec import SystemSpec

@dataclass(frozen=True)
class CartPoleSpec(SystemSpec):
    system_name: str
    system_type: str
    cart_mass: float
    pole_mass: float
    pole_length: float
    gravity: float
    state_dim: int
    input_dim: int
    state_names: list[str]
    input_names: list[str]
    ref_names: list[str]
    system_dim: int

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CartPoleSpec":
        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        return cls.from_dict(cfg)

    @classmethod
    def from_dict(cls, cfg: dict) -> "CartPoleSpec":
        if cfg.get("system_type") != "cartpole":
            raise ValueError(
                f"CartPoleSpec expects system_type='cartpole', "
                f"got {cfg.get('system_type')!r}."
            )

        return cls(**cfg)

    def __post_init__(self):
        if self.cart_mass <= 0:
            raise ValueError(f"cart_mass must be > 0, got {self.cart_mass}")

        if self.pole_mass <= 0:
            raise ValueError(f"pole_mass must be > 0, got {self.pole_mass}")

        if self.pole_length <= 0:
            raise ValueError(f"pole_length must be > 0, got {self.pole_length}")

        if self.gravity <= 0:
            raise ValueError(f"gravity must be > 0, got {self.gravity}")

    @property
    def x_dim(self) -> int:
        x_dim, _, _ = get_cartpole_dimensions()
        return x_dim

    @property
    def u_dim(self) -> int:
        _, u_dim, _ = get_cartpole_dimensions()
        return u_dim

    @property
    def x_ref_dim_closed_loop(self) -> int:
        _, _, x_ref_dim = get_cartpole_dimensions(task="control")
        return x_ref_dim

    @property
    def x_ref_dim_open_loop(self) -> int:
        _, _, x_ref_dim = get_cartpole_dimensions(task="open_loop")
        return x_ref_dim

    @property
    def num_views(self) -> int:
        return get_num_views()

    def get_x_labels(self, only_positions: bool = False) -> List[str]:
        return get_x_labels(only_positions)

    def get_x_names(self, only_positions: bool = False) -> List[str]:
        return get_x_names(only_positions)

    def get_u_labels(self) -> List[str]:
        return get_u_labels()

    @property
    def angle_indexes(self) -> List[int]:
        return get_angle_indexes()

    def convert_state_to_deg(self, x: np.ndarray) -> np.ndarray:
        return convert_rad_to_deg_np(x, self.angle_indexes)

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

    def __repr__(self) -> str:
        return (
            f"CartPoleSpec(name={self.system_name!r}, "
            f"x_dim={self.x_dim}, u_dim={self.u_dim})"
        )

    def get_input_plot_groups(self, group_inputs: bool = True) -> list[dict]:
        labels = self.get_u_labels()

        return [
            {
                "indices": [0],
                "label": labels[0],
            }
        ]