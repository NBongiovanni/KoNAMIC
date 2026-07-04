from __future__ import annotations

from typing import Any

from KoNAMIC import config, paths
from KoNAMIC.core.models import ModelConfig
from KoNAMIC.core.systems import SystemSpec
from KoNAMIC.pipelines.data_preparation import SensorPreparationConfig


def load_open_loop_run_configs(
    *,
    run_paths: paths.RunPaths,
    system_spec: SystemSpec,
) -> tuple[ModelConfig, SensorPreparationConfig]:
    run_config = load_stored_run_config(run_paths)

    model_raw = require_first_present(
        run_config,
        keys=("model", "model_params"),
        context=str(run_paths.run_dir),
    )
    data_preparation_raw = require_first_present(
        run_config,
        keys=("data_preparation", "dataset_params"),
        context=str(run_paths.run_dir),
    )

    data_preparation_config = SensorPreparationConfig.from_dict(data_preparation_raw)
    model_config = ModelConfig.from_dict(model_raw)
    model_config = model_config.with_system_dimensions(
        x_dim=system_spec.x_dim,
        u_dim=system_spec.u_dim,
    ).with_delay(data_preparation_config.postprocessing.delay)
    return model_config, data_preparation_config


def load_stored_run_config(run_paths: paths.RunPaths) -> dict[str, Any]:
    config_path = run_paths.run_dir / "config.yaml"
    if config_path.exists():
        return config.load_yaml(config_path)

    legacy_config_path = run_paths.run_dir / "sensor_3d.yaml"
    if legacy_config_path.exists():
        return config.load_checkpoint_config(run_paths.run_dir)

    raise FileNotFoundError(
        "No run config found for open-loop evaluation. Expected either "
        f"{config_path} or {legacy_config_path}."
    )


def require_first_present(
    cfg: dict[str, Any],
    *,
    keys: tuple[str, ...],
    context: str,
):
    for key in keys:
        if key in cfg:
            return cfg[key]

    raise KeyError(
        f"Missing one of required keys {keys} in {context}. "
        f"Available keys: {list(cfg.keys())}"
    )
