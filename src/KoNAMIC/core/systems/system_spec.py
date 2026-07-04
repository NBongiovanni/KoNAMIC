from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class SystemSpec:
    """
    Base class for physical system specifications.

    This class contains the interface expected by generic modules such as
    visualization, simulation, and evaluation.
    """

    system_name: str
    system_type: str
    state_dim: int
    input_dim: int
    state_names: list[str]
    input_names: list[str]
    ref_names: list[str]
    system_dim: int

    @classmethod
    def from_yaml(cls, path: str | Path):
        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if not isinstance(cfg, dict):
            raise TypeError(
                f"Expected YAML file {path} to contain a dictionary, "
                f"got {type(cfg).__name__}."
            )

        return cls.from_dict(cfg)

    @classmethod
    def from_dict(cls, cfg: dict):
        return cls(**cfg)

    @property
    def x_dim(self) -> int:
        return self.state_dim

    @property
    def u_dim(self) -> int:
        return self.input_dim

    @property
    def x_ref_dim_closed_loop(self) -> int:
        return self.state_dim

    @property
    def x_ref_dim_open_loop(self) -> int:
        return self.state_dim

    @property
    def num_views(self) -> int:
        return 1

    @property
    def angle_indexes(self) -> list[int]:
        return []

    def get_x_labels(self, only_positions: bool = False) -> list[str]:
        if only_positions:
            return self.state_names[: self.state_dim // 2]
        return list(self.state_names)

    def get_x_names(self, only_positions: bool = False) -> list[str]:
        if only_positions:
            return self.state_names[: self.state_dim // 2]
        return list(self.state_names)

    def get_u_labels(self) -> list[str]:
        return list(self.input_names)

    def split_state(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x)
        half = self.x_dim // 2
        return x[..., :half], x[..., half:]

    def convert_state_to_deg(self, x: np.ndarray) -> np.ndarray:
        return self.convert_available_angles_to_deg(x)

    def convert_available_angles_to_deg(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).copy()

        for idx in self.angle_indexes:
            if idx < x.shape[-1]:
                x[..., idx] = np.rad2deg(x[..., idx])

        return x

    def check_state_dim(self, x: np.ndarray) -> None:
        if x.shape[-1] != self.x_dim:
            raise ValueError(
                f"Expected state dim {self.x_dim}, got {x.shape[-1]}"
            )

    def check_control_dim(self, u: np.ndarray) -> None:
        if u.shape[-1] != self.u_dim:
            raise ValueError(
                f"Expected control dim {self.u_dim}, got {u.shape[-1]}"
            )

    def get_input_plot_groups(self, group_inputs: bool = True) -> list[dict]:
        labels = self.get_u_labels()

        return [
            {
                "indices": [i],
                "label": labels[i],
            }
            for i in range(self.u_dim)
        ]