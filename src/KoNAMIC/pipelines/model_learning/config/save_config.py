#!/usr/bin/env python
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")

from KoNAMIC import config, paths
from .pipeline_config import TrainingPipelineConfig


def save_effective_run_config(
    *,
    run_config: TrainingPipelineConfig,
    run_paths: paths.RunPaths,
    args,
    run_stamp: str,
    run_status: str,
) -> None:
    config_blocks = run_config.to_dict()
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_stamp": run_stamp,
        "run_status": run_status,
        "system_name": args.system_name,
        "modality": args.modality,
        "dataset_stamp": str(args.dataset_stamp),
        "cli_args": vars(args),
    }
    full_config = {
        "metadata": metadata,
        **config_blocks,
    }

    config.save_yaml(full_config, "config.yaml", run_paths.run_dir)

    split_config_dir = run_paths.run_dir / "configs"
    config.save_yaml(metadata, "metadata.yaml", split_config_dir)
    for block_name, block in config_blocks.items():
        config.save_yaml(block, f"{block_name}.yaml", split_config_dir)
