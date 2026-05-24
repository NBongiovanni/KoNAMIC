from __future__ import annotations
import random
from typing import Tuple

import numpy as np
import torch

from KoNAMIC.core import drone, utils
from KoNAMIC.pipelines.data_pipeline import StateInputsDatasetBuilder
from KoNAMIC.core.models import load_sensor_koop_model_for_eval, SensorValForwardOutputs

from .trajectories import OpenLoopSensorResult


def open_loop_simulation_sensor_pipeline(
    case: utils.CaseConfig,
    phase: str,
    num_steps: int,
    modality: str,
    drone_dim: int,
    stamp_open_loop: str,
    seed: int,
) -> OpenLoopSensorResult:
    """Perform open-loop forward simulation (sensor modality)."""

    _set_seed(seed)

    # --- Paths & config ---
    paths, params = _prepare_paths_and_params(
        case,
        modality,
        drone_dim,
        stamp_open_loop,
        seed
    )

    device = utils.load_device()

    # --- Dataset ---
    data_loader = _build_dataloader(params, drone_dim)

    # --- Model ---
    koop_model, x_scaler, u_scaler = _load_model(case, paths, params)

    A, B = koop_model.construct_koop_matrices()
    print(f"A:\n{A}")
    print(f"B:\n{B}")

    # --- Forward ---
    with torch.no_grad():
        x_gt_scaled, u_traj = _get_one_batch(data_loader[phase])

        rec, pred = koop_model.forward(
            x_gt_scaled[:, 0].to(device),
            u_traj.to(device),
            num_steps,
        )

        outputs = _postprocess_outputs(
            koop_model,
            rec,
            pred,
            x_gt_scaled,
            u_traj,
            x_scaler,
            u_scaler,
            drone_dim,
            device
        )

    return OpenLoopSensorResult(
        val_output=outputs,
        u_scaler=u_scaler,
        x_scaler=x_scaler,
        run_dir=paths.run_dir,
        open_loop_eval_dir=paths.run_dir / "checkpoints" / "open_loop",
    )


# ============================
# Étapes du pipeline
# ============================

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def _prepare_paths_and_params(
    case,
    modality: str,
    drone_dim: int,
    stamp_open_loop: str,
    seed: int,
) -> tuple:
    paths = utils.build_run_paths(
        modality,
        drone_dim,
        case.run_status,
        case.stamp,
        stamp_open_loop
    )
    params = utils.load_checkpoint_config(paths)

    params["seed"] = seed
    return paths, params


def _build_dataloader(params: dict, drone_dim: int):
    dataset_params = params["dataset_params"]
    builder = StateInputsDatasetBuilder(dataset_params, drone_dim)
    return builder.data_loader


def _load_model(case, paths, params: dict):
    model_params = params["model_params"]

    return load_sensor_koop_model_for_eval(
        model_params,
        case.epoch,
        paths.run_dir,
    )


def _postprocess_outputs(
    koop_model,
    rec,
    pred,
    x_gt_scaled,
    u_traj,
    x_scaler,
    u_scaler,
    drone_dim,
    device
):
    # --- Unscale ---
    rec_phys = _unscale(rec, x_scaler)
    x_gt_phys = _unscale(x_gt_scaled, x_scaler)
    pred_x_phys = _unscale(pred.state, x_scaler)
    u_traj_phys = _to_tensor(_unscale(u_traj, u_scaler))

    # --- Angles ---
    rec_deg = _convert_angles_deg(rec_phys, drone_dim)
    pred.state = _convert_angles_deg(pred_x_phys, drone_dim)
    x_gt_phys_deg = _convert_angles_deg(x_gt_phys, drone_dim)

    # --- Projection ---
    z_proj = koop_model.batch_projection(x_gt_scaled.to(device))

    return SensorValForwardOutputs(
        rec=rec_deg,
        pred=pred,
        proj=z_proj,
        state_gt_scaled=x_gt_scaled,
        state_gt_physical=x_gt_phys_deg,
        inputs_scaled=u_traj,
        inputs_physical=u_traj_phys
    )


# ============================
# Utils internes
# ============================

def _get_one_batch(dataloader: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    try:
        return next(iter(dataloader))
    except StopIteration as e:
        ds_len = len(dataloader.dataset) if hasattr(dataloader, "dataset") else "unknown"
        bs = getattr(dataloader, "batch_size", "unknown")
        drop_last = getattr(dataloader, "drop_last", "unknown")

        raise RuntimeError(
            f"DataLoader is empty: len(dataset)={ds_len}, batch_size={bs}, drop_last={drop_last}."
        ) from e


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _to_tensor(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(a).float()


def _unscale(t: torch.Tensor, scaler) -> np.ndarray:
    if scaler is None:
        return _to_numpy(t)

    x = _to_numpy(t)

    if x.ndim == 3:
        b, t, f = x.shape
        return scaler.inverse_transform(x.reshape(-1, f)).reshape(b, t, f)

    if x.ndim == 2:
        return scaler.inverse_transform(x)

    if x.ndim == 1:
        return scaler.inverse_transform(x.reshape(1, -1)).reshape(-1)

    raise ValueError(f"Unexpected shape: {x.shape}")


def _convert_angles_deg(a: np.ndarray, drone_dim: int) -> torch.Tensor:
    angles_indexes = drone.get_angle_indexes(drone_dim)
    return _to_tensor(drone.convert_rad_to_deg_np(a, angles_indexes))