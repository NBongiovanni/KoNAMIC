from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import random
import numpy as np
import torch
from torch import Tensor

from KoNAMIC.core import utils
from KoNAMIC.pipelines.data_pipeline import ImageDatasetBuilder, StateInputsDatasetBuilder
from KoNAMIC.core.models import load_vision_koop_model_for_eval, VisionKoopModel
from KoNAMIC.pipelines.model_learning import build_ground_truth_from_images
from ..vision_postprocessing import sim_output_processing_vision

# -----------------------------------------------------------------------------
# Internal context to keep the pipeline readable
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class _OpenLoopVisionContext:
    paths: Any
    params: Dict[str, Any]
    koop_model: VisionKoopModel
    u_scaler: Any
    training_params: dict
    im_data_loader: Dict[str, Any]
    device: torch.device


def open_loop_simulation_vision_pipeline(
        case: utils.CaseConfig,
        phase: str,
        modality: str,
        num_steps: int,
        seed: int,
        stamp_open_loop: str,
        dt: float,
        dataset_version: int,
):
    """Perform open-loop forward simulation and return everything needed for visualization."""
    _set_seeds(seed)

    paths, params = _prepare_paths_and_params(
        case=case,
        modality=modality,
        num_steps_pred=num_steps,
        seed=seed,
        stamp_open_loop=stamp_open_loop,
        dataset_version=dataset_version,
    )

    koop_model, u_scaler, training_params = _load_model_and_scalers(
        params=params,
        case=case,
        run_dir=paths.run_dir,
    )

    dataset_params = params["dataset_params"]
    im_data_loader = _build_dataloader(dataset_params, case.drone_dim)

    device = next(koop_model.parameters()).device
    ctx = _OpenLoopVisionContext(
        paths=paths,
        params=params,
        koop_model=koop_model,
        u_scaler=u_scaler,
        training_params=training_params,
        im_data_loader=im_data_loader,
        device=device,
    )

    out, y_data, u_data, x_data = _run_open_loop_forward(ctx, phase, num_steps)
    ground_truth = build_ground_truth_from_images(
        y_data,
        x_data,
        case.drone_dim,
        2,
    )

    val_output = sim_output_processing_vision(
        output=out,
        ground_truth=ground_truth,
        u_gt=u_data,
        x_gt=x_data,
        u_scaler=ctx.u_scaler,
        drone_dim=case.drone_dim,
        dt=dt,
    )

    return OpenLoopVisionResult(
        val_output=val_output,
        u_scaler=ctx.u_scaler,
        training_params=ctx.training_params,
        run_dir=ctx.paths.run_dir,
    )

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_paths_and_params(
        case: utils.CaseConfig,
        modality: str,
        num_steps_pred: int,
        seed: int,
        stamp_open_loop: str,
        dataset_version: int,
) -> Tuple[Any, Dict[str, Any]]:
    paths = utils.build_run_paths(
        modality,
        case.drone_dim,
        case.run_status,
        case.stamp,
        stamp_open_loop,
    )
    params = utils.load_checkpoint_config(paths)
    params = utils.process_checkpoint_config(
        params,
        case.dynamics,
        case.geom_losses,
        paths,
        seed
    )

    # Dataset params overrides
    dataset_params = params["dataset_params"]
    dataset_params["dataset_version"] = dataset_version
    dataset_params["val_datasets"][1]["num_steps_pred"] = num_steps_pred
    return paths, params


def _load_model_and_scalers(
        params: Dict[str, Any],
        case: utils.CaseConfig,
        run_dir: Path,
) -> Tuple[VisionKoopModel, Any, dict]:
    koop_model, _x_scaler, u_scaler = load_vision_koop_model_for_eval(
        params["model_params"],
        case.epoch,
        run_dir,
    )
    koop_model.eval()

    training_params = params["training_params"]
    return koop_model, u_scaler, training_params


def _build_dataloader(dataset_params: Dict[str, Any], drone_dim: int) -> Dict[str, Any]:
    sensor_builder = StateInputsDatasetBuilder(dataset_params, drone_dim)
    processed_sensor_data = sensor_builder.processed

    im_dataset_builder = ImageDatasetBuilder(
        dataset_version=dataset_params["dataset_version"],
        num_val_datasets=2,
        resolution=dataset_params["resolution"],
        batch_size=dataset_params["batch_size"],
        processed_dataset=processed_sensor_data,
        num_workers=0,
        drone_dim=drone_dim,
        seed=0,
    )
    return im_dataset_builder.pipeline()


def _run_open_loop_forward(
        ctx: _OpenLoopVisionContext,
        phase: str,
        num_steps: int,
) -> Tuple[Any, Tensor, Tensor, Tensor]:
    y_data, u_data, x_data = _get_one_batch(ctx.im_data_loader[phase])

    y_data = y_data.to(ctx.device)
    u_data = u_data.to(ctx.device)
    x_data = x_data.to(ctx.device)

    with torch.no_grad():
        out = ctx.koop_model.forward(y_data[:, 0], u_data, num_steps)
    return out, y_data, u_data, x_data


def _get_one_batch(dataloader: Any):
    return next(iter(dataloader))