from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch

from KoNAMIC.koopman.models import (
    ModelConfig,
    SensorValForwardOutputs,
    VisionValForwardOutputs,
    load_sensor_koop_model_for_eval,
)
from KoNAMIC.koopman.models.outputs.vision_outputs import GroundTruth
from KoNAMIC.core.rendering.features.geometric_features_robust import (
    compute_angles_robust,
    compute_centroids_robust,
)
from KoNAMIC.core.systems import SystemSpec
from KoNAMIC.paths import DatasetPaths, RunPaths
from KoNAMIC.pipelines.data_preparation import SensorBuilder, SensorPreparationConfig


@torch.no_grad()
def make_sensor_rollout_output(
    *,
    koop_model,
    dataloader: torch.utils.data.DataLoader,
    x_scaler,
    u_scaler,
    system_spec: SystemSpec,
    num_steps: int,
    device,
) -> SensorValForwardOutputs:
    x_gt_scaled, u_traj = get_one_batch(dataloader)

    rec, pred = koop_model.forward(
        x_gt_scaled[:, 0].to(device),
        u_traj.to(device),
        num_steps,
    )

    return postprocess_outputs(
        koop_model=koop_model,
        rec=rec,
        pred=pred,
        x_gt_scaled=x_gt_scaled,
        u_traj=u_traj,
        x_scaler=x_scaler,
        u_scaler=u_scaler,
        system_spec=system_spec,
        device=device,
    )


@torch.no_grad()
def make_vision_rollout_output(
    *,
    koop_model,
    dataloader: torch.utils.data.DataLoader,
    u_scaler,
    num_steps: int,
    device,
) -> VisionValForwardOutputs:
    y_traj, u_traj, x_data = get_one_vision_batch(dataloader)
    _validate_vision_rollout_horizon(
        num_steps=num_steps,
        y_traj=y_traj,
        u_traj=u_traj,
        x_data=x_data,
    )

    y_traj_device = y_traj[:, :num_steps].to(device)
    u_traj_device = u_traj[:, :num_steps].to(device)
    x_data = x_data[:, :num_steps]

    outputs = koop_model.forward(
        y_init=y_traj_device[:, 0],
        u_traj=u_traj_device,
        num_steps=num_steps,
    )
    z_proj = koop_model.batch_projection(y_traj_device, u_traj_device)

    targets = build_vision_ground_truth(
        y_traj=y_traj_device,
        x_data=x_data.to(device),
        num_views=koop_model.num_views,
    )

    pred_horizon = outputs.pred.z.shape[1]
    outputs.pred.state = outputs.pred.build_state_left()
    outputs.pred.state_right = (
        outputs.pred.build_state_right()
        if koop_model.num_views == 2
        else None
    )
    targets.state = _build_vision_state(
        targets.centroids_left,
        targets.angles_left,
    )[:, 1:pred_horizon + 1]
    targets.state_right = (
        _build_vision_state(targets.centroids_right, targets.angles_right)[:, 1:pred_horizon + 1]
        if koop_model.num_views == 2
        else None
    )

    u_traj_aligned = u_traj[:, :pred_horizon]
    u_traj_phys = _to_tensor(_unscale(u_traj_aligned, u_scaler))

    return VisionValForwardOutputs(
        rec=outputs.rec,
        pred=outputs.pred,
        g_t=targets,
        inputs_scaled=u_traj_aligned,
        inputs_physical=u_traj_phys,
        state=targets.state,
    )


def _validate_vision_rollout_horizon(
    *,
    num_steps: int,
    y_traj: torch.Tensor,
    u_traj: torch.Tensor,
    x_data: torch.Tensor,
) -> None:
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}.")

    lengths = {
        "vision sequence": y_traj.shape[1],
        "input sequence": u_traj.shape[1],
        "state sequence": x_data.shape[1],
    }
    too_short = {name: length for name, length in lengths.items() if length < num_steps}
    if too_short:
        available = ", ".join(
            f"{name}={length}" for name, length in too_short.items()
        )
        raise ValueError(
            f"Requested vision rollout with num_steps={num_steps}, but the "
            f"batch only provides {available}. Increase num_steps_pred for the "
            "evaluated split in the data_preparation config, or reduce "
            "open_loop.num_steps_simulation."
        )


def build_vision_ground_truth(
    *,
    y_traj: torch.Tensor,
    x_data: torch.Tensor,
    num_views: int,
) -> GroundTruth:
    current_views = _extract_current_views(y_traj, num_views)
    y_left = current_views[:, :, 0:1]

    if num_views == 2:
        y_right = current_views[:, :, 1:2]
    elif num_views == 1:
        y_right = torch.zeros_like(y_left)
    else:
        raise ValueError(f"Invalid num_views={num_views}")

    return GroundTruth(
        y_left=y_left,
        y_right=y_right,
        centroids_left=compute_centroids_robust(y_left),
        angles_left=compute_angles_robust(y_left),
        centroids_right=compute_centroids_robust(y_right),
        angles_right=compute_angles_robust(y_right),
        x_data=x_data,
    )


def _extract_current_views(y_traj: torch.Tensor, num_views: int) -> torch.Tensor:
    if y_traj.dim() != 5:
        raise ValueError(f"Expected y_traj as (B,T,C,H,W), got {tuple(y_traj.shape)}")
    if y_traj.shape[2] < num_views:
        raise ValueError(
            f"Expected at least {num_views} image channels, got {y_traj.shape[2]}."
        )
    return y_traj[:, :, -num_views:]


def _build_vision_state(centroids: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    if centroids.ndim == 4 and centroids.shape[-2] == 1:
        centroids = centroids.squeeze(-2)
    if angles.ndim == centroids.ndim - 1:
        angles = angles.unsqueeze(-1)
    if angles.ndim == 4 and angles.shape[-2] == 1 and angles.shape[-1] == 1:
        angles = angles.squeeze(-2)
    return torch.cat([centroids, angles], dim=-1)


def postprocess_outputs(
    *,
    koop_model,
    rec,
    pred,
    x_gt_scaled,
    u_traj,
    x_scaler,
    u_scaler,
    system_spec: SystemSpec,
    device,
):
    horizon = pred.state.shape[1]
    x_gt_scaled = x_gt_scaled[:, :horizon]
    u_traj = u_traj[:, :horizon]

    rec_phys = _unscale(rec, x_scaler)
    x_gt_phys = _unscale(x_gt_scaled, x_scaler)
    pred_x_phys = _unscale(pred.state, x_scaler)
    u_traj_phys = _to_tensor(_unscale(u_traj, u_scaler))

    rec_deg = _convert_angles_deg(rec_phys, system_spec)
    pred.state = _convert_angles_deg(pred_x_phys, system_spec)
    x_gt_phys_deg = _convert_angles_deg(x_gt_phys, system_spec)

    z_proj = koop_model.batch_projection(x_gt_scaled.to(device))

    return SensorValForwardOutputs(
        rec=rec_deg,
        pred=pred,
        proj=z_proj,
        state_gt_scaled=x_gt_scaled,
        state_gt_physical=x_gt_phys_deg,
        inputs_scaled=u_traj,
        inputs_physical=u_traj_phys,
    )



def get_one_vision_batch(
    dataloader: torch.utils.data.DataLoader,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    first_batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    for batch in dataloader:
        y_batch, _, _ = batch
        if first_batch is None:
            first_batch = batch

        if _has_state_activity(y_batch):
            return batch

    if first_batch is not None:
        return first_batch

    ds_len = len(dataloader.dataset) if hasattr(dataloader, "dataset") else "unknown"
    bs = getattr(dataloader, "batch_size", "unknown")
    drop_last = getattr(dataloader, "drop_last", "unknown")

    raise RuntimeError(
        f"Vision DataLoader is empty: len(dataset)={ds_len}, batch_size={bs}, "
        f"drop_last={drop_last}."
    )


def get_one_batch(
    dataloader: torch.utils.data.DataLoader,
) -> Tuple[torch.Tensor, torch.Tensor]:
    first_batch: Tuple[torch.Tensor, torch.Tensor] | None = None

    for batch in dataloader:
        x_batch, _ = batch
        if first_batch is None:
            first_batch = batch

        if _has_state_activity(x_batch):
            return batch

    if first_batch is not None:
        return first_batch

    ds_len = len(dataloader.dataset) if hasattr(dataloader, "dataset") else "unknown"
    bs = getattr(dataloader, "batch_size", "unknown")
    drop_last = getattr(dataloader, "drop_last", "unknown")

    raise RuntimeError(
        f"DataLoader is empty: len(dataset)={ds_len}, batch_size={bs}, "
        f"drop_last={drop_last}."
    )


def _has_state_activity(x_batch: torch.Tensor, *, atol: float = 1e-8) -> bool:
    return bool(torch.max(torch.abs(x_batch)).item() > atol)


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _to_tensor(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(a).float()


def _unscale(t: torch.Tensor, scaler) -> np.ndarray:
    if scaler is None:
        return _to_numpy(t)

    x = _to_numpy(t)

    if x.ndim == 3:
        b, n_steps, f = x.shape
        return scaler.inverse_transform(x.reshape(-1, f)).reshape(b, n_steps, f)

    if x.ndim == 2:
        return scaler.inverse_transform(x)

    if x.ndim == 1:
        return scaler.inverse_transform(x.reshape(1, -1)).reshape(-1)

    raise ValueError(f"Unexpected shape: {x.shape}")


def _convert_angles_deg(a: np.ndarray, system_spec: SystemSpec) -> torch.Tensor:
    return _to_tensor(system_spec.convert_available_angles_to_deg(a))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_dataloader(
    *,
    dataset_paths: DatasetPaths,
    data_preparation_config: SensorPreparationConfig,
    system_spec: SystemSpec,
):
    builder = SensorBuilder(
        dataset_paths,
        data_preparation_config,
        system_spec.system_dim,
    )
    return builder.data_loaders


def _load_model(
    *,
    model_config: ModelConfig,
    epoch: int,
    run_paths: RunPaths,
):
    return load_sensor_koop_model_for_eval(
        model_config,
        epoch,
        run_paths.run_dir,
    )
