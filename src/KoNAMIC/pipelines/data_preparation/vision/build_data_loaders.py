from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .config import SampleConfig, ValDatasetConfig, VisionPreparationConfig
from .vision_processor import VisionProcessor


def iter_sample_configs(dataset_config: VisionPreparationConfig) -> list[dict[str, Any]]:
    train_cfg = _sample_to_dict(dataset_config.train)
    return [
        {
            "split": "train",
            "name": "train",
            **train_cfg,
        },
        *[_sample_to_dict(val_cfg) for val_cfg in dataset_config.val_datasets],
    ]


def _sample_to_dict(sample_cfg: SampleConfig | ValDatasetConfig) -> dict[str, Any]:
    return asdict(sample_cfg)


def prepare_vision_dataset(
    data_preparation_config: VisionPreparationConfig,
    dataset_stamp: str,
) -> None:

    for sample_cfg in iter_sample_configs(data_preparation_config):
        phase = sample_cfg["split"]

        num_traj_loaded = sample_cfg["num_traj_loaded"]
        num_steps_loaded = sample_cfg["num_steps_loaded"]
        num_steps_pred = sample_cfg["num_steps_pred"]

        print(
            f"[INFO] Preparing vision memmap for {phase}: "
            f"num_traj_loaded={num_traj_loaded}, "
            f"num_steps_loaded={num_steps_loaded}, "
            f"num_steps_pred={num_steps_pred}"
        )

        vision_processor = VisionProcessor(
            data_preparation_config,
            phase,
            dataset_stamp,
            num_steps_loaded,
        )

        vision_processor.pipeline(
            num_traj_loaded,
            data_preparation_config.resolution,
            num_steps_pred,
        )
