from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataLoaderConfig:
    batch_size: int = 128
    num_workers: int = 2


@dataclass
class PostprocessingConfig:
    ds_step: int = 10
    smooth_window: int = 10
    delay: int = 0


@dataclass
class ScalerConfig:
    scale_x: bool = True
    scale_u: bool = True
    center: bool = True
    mean_x: list[float] = field(default_factory=list)
    mean_u: list[float] = field(default_factory=list)


@dataclass
class SampleConfig:
    num_traj_loaded: int
    num_steps_loaded: int = 300
    num_steps_pred: int = 50


@dataclass
class ValDatasetConfig(SampleConfig):
    split: str = "val_1"
    name: str = "short_horizon"


@dataclass
class SensorPreparationConfig:
    drone_dim: int = 2

    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)

    train: SampleConfig = field(default_factory=SampleConfig)
    val_datasets: list[ValDatasetConfig] = field(default_factory=list)

    scaler: ScalerConfig = field(default_factory=ScalerConfig)

    @property
    def batch_size(self) -> int:
        return self.dataloader.batch_size

    @property
    def num_workers(self) -> int:
        return self.dataloader.num_workers

    @property
    def ds_step(self) -> int:
        return self.postprocessing.ds_step

    @property
    def smooth_window(self) -> int:
        return self.postprocessing.smooth_window

    @property
    def delay(self) -> int:
        return self.postprocessing.delay

    @classmethod
    def from_yaml(cls, path: Path) -> "SensorPreparationConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "SensorPreparationConfig":
        dataloader = cfg.get("dataloader", {})
        postprocessing = cfg.get("postprocessing", {})
        train = cfg.get("train", {})
        scaler = cfg.get("scaler", {})

        return cls(
            drone_dim=int(cfg.get("drone_dim", 2)),

            dataloader=DataLoaderConfig(
                batch_size=int(dataloader.get("batch_size", 128)),
                num_workers=int(dataloader.get("num_workers", 2)),
            ),

            postprocessing=PostprocessingConfig(
                ds_step=int(postprocessing.get("ds_step", 10)),
                smooth_window=int(postprocessing.get("smooth_window", 10)),
                delay=int(postprocessing.get("delay", 0)),
            ),

            train=SampleConfig(
                num_traj_loaded=int(train["num_traj_loaded"]),
                num_steps_loaded=int(train.get("num_steps_loaded", 300)),
                num_steps_pred=int(train.get("num_steps_pred", 50)),
            ),

            val_datasets=[
                ValDatasetConfig(
                    split=val_cfg.get("split", f"val_{idx + 1}"),
                    name=val_cfg.get("name", f"val_{idx + 1}"),
                    num_traj_loaded=int(val_cfg["num_traj_loaded"]),
                    num_steps_loaded=int(val_cfg.get("num_steps_loaded", 300)),
                    num_steps_pred=int(val_cfg.get("num_steps_pred", 50)),
                )
                for idx, val_cfg in enumerate(cfg.get("val_datasets", []))
            ],

            scaler=ScalerConfig(
                scale_x=bool(scaler.get("scale_x", True)),
                scale_u=bool(scaler.get("scale_u", True)),
                center=bool(scaler.get("center", True)),
                mean_x=list(scaler.get("mean_x", [])),
                mean_u=list(scaler.get("mean_u", [])),
            ),
        )

    def to_dict(self) -> dict:
        return {
            "drone_dim": self.drone_dim,
            "dataloader": {
                "batch_size": self.dataloader.batch_size,
                "num_workers": self.dataloader.num_workers,
            },
            "postprocessing": {
                "ds_step": self.postprocessing.ds_step,
                "smooth_window": self.postprocessing.smooth_window,
                "delay": self.postprocessing.delay,
            },
            "train": {
                "num_traj_loaded": self.train.num_traj_loaded,
                "num_steps_loaded": self.train.num_steps_loaded,
                "num_steps_pred": self.train.num_steps_pred,
            },
            "val_datasets": [
                {
                    "split": val.split,
                    "name": val.name,
                    "num_traj_loaded": val.num_traj_loaded,
                    "num_steps_loaded": val.num_steps_loaded,
                    "num_steps_pred": val.num_steps_pred,
                }
                for val in self.val_datasets
            ],
            "scaler": {
                "scale_x": self.scaler.scale_x,
                "scale_u": self.scaler.scale_u,
                "center": self.scaler.center,
                "mean_x": self.scaler.mean_x,
                "mean_u": self.scaler.mean_u,
            },
        }