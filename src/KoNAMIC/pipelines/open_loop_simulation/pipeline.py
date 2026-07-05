from __future__ import annotations

from KoNAMIC import utils
from KoNAMIC.koopman.models import ModelConfig
from KoNAMIC.core.systems import SystemSpec
from KoNAMIC.paths import DatasetPaths, RunPaths
from KoNAMIC.pipelines.data_preparation import SensorPreparationConfig

from .trajectories import OpenLoopSensorResult
from .post_process import (
    make_sensor_rollout_output,
    _build_dataloader,
    _load_model,
    _set_seed,
)


def open_loop_simulation_sensor_pipeline(
    *,
    run_paths: RunPaths,
    dataset_paths: DatasetPaths,
    model_config: ModelConfig,
    data_preparation_config: SensorPreparationConfig,
    system_spec: SystemSpec,
    phase: str,
    num_steps: int,
    epoch: int,
    seed: int,
) -> OpenLoopSensorResult:
    """Perform open-loop forward simulation for the sensor modality."""

    _set_seed(seed)
    device = utils.load_device()

    data_loader = _build_dataloader(
        dataset_paths=dataset_paths,
        data_preparation_config=data_preparation_config,
        system_spec=system_spec,
    )
    koop_model, x_scaler, u_scaler = _load_model(
        model_config=model_config,
        epoch=epoch,
        run_paths=run_paths,
    )

    outputs = make_sensor_rollout_output(
        koop_model=koop_model,
        dataloader=data_loader[phase],
        x_scaler=x_scaler,
        u_scaler=u_scaler,
        system_spec=system_spec,
        num_steps=num_steps,
        device=device,
    )

    return OpenLoopSensorResult(
        val_output=outputs,
        u_scaler=u_scaler,
        x_scaler=x_scaler,
        run_dir=run_paths.run_dir,
        open_loop_eval_dir=run_paths.open_loop_eval_dir,
    )


