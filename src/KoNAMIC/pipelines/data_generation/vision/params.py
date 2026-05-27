from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

import yaml

@dataclass
class VisionDatasetParams:
    im_size: int
    resolution: int
    drone_dim: int

    dt: float
    batch_size: int

    train: dict
    val_datasets: dict

    downsample_factor: int
    scaler: dict
    num_workers: int = 0
    delay: int = 1

    image_dataset_dir: Optional[Path] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "VisionDatasetParams":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VisionDatasetParams":
        return cls(
            im_size=int(data["im_size"]),
            resolution=int(data["resolution"]),
            drone_dim=int(data["drone_dim"]),
            dt=float(data["dt"]),
            batch_size=int(data["batch_size"]),
            train=data["train"],
            val_datasets=data["val_datasets"],
            downsample_factor=int(data.get("downsample_factor", 1)),
            scaler=data["scaler"],
            num_workers=int(data.get("num_workers", 0)),
            delay=int(data.get("delay", 1)),
        )