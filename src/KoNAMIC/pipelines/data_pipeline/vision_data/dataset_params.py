# KSIC_v6/config/dataset_params.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

import yaml

@dataclass
class DatasetParams:
    # --- Champs directement issus du YAML (racine) ---
    im_size: int
    resolution: int
    drone_dim: int

    dataset_version: int
    dt: float
    batch_size: int

    train: dict                 # <-- restera "train" pour l’instant
    val_datasets: dict

    decimation_factor: int
    scaler: dict
    num_workers: int = 0
    delay: int = 1

    image_dataset_dir: Optional[Path] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "DatasetParams":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetParams":
        return cls(
            im_size=int(data["im_size"]),
            resolution=int(data["resolution"]),
            drone_dim=int(data["drone_dim"]),
            dataset_version=int(data["dataset_version"]),
            dt=float(data["dt"]),
            batch_size=int(data["batch_size"]),
            train=data["train"],
            val_datasets=data["val_datasets"],
            decimation_factor=int(data.get("decimation_factor", 1)),
            scaler=data["scaler"],
            num_workers=int(data.get("num_workers", 0)),
            delay=int(data.get("delay", 1)),
        )