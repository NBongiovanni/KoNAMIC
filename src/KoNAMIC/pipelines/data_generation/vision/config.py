from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

@dataclass
class VisionGenerationConfig:
    resolution: int
    drone_dim: int
    render_stride: int
    image_dataset_dir: Path | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> "VisionGenerationConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VisionGenerationConfig":
        return cls(
            resolution=int(data["resolution"]),
            drone_dim=int(data["drone_dim"]),
            render_stride=int(data["render_stride"]),
        )