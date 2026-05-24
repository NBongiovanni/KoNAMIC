from __future__ import annotations

from dataclasses import replace
from typing import Literal, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from sklearn.preprocessing import StandardScaler

from KoNAMIC.core import utils
from KoNAMIC.core.models import VisionValForwardOutputs, ForwardOutputs, GroundTruth

RAD_TO_DEG = 180.0 / np.pi

# =============================================================================
# Config / conventions
# =============================================================================
PadMode = Literal["zeros", "replicate"]

@dataclass(frozen=True)
class VisionProcessingConfig:
    drone_dim: int
    dt: float

    # geometry
    img_size: int = 128
    world_size: float = 1

    # angle convention
    output_angles_in_deg: bool = True
    # if your angle is computed in image coordinates (y down) but you want physical (y up)
    flip_angle_sign: bool = False

    # finite difference
    pad_mode: PadMode = "zeros"

    # which "point" to use if you have multiple detected points
    point_idx: int = 0

# =============================================================================
# Public entrypoint
# =============================================================================

def sim_output_processing_vision(
        output: ForwardOutputs,
        ground_truth: GroundTruth,
        u_gt: Tensor,
        x_gt: Tensor,
        u_scaler: StandardScaler,
        drone_dim: int,
        dt: float,
) -> VisionValForwardOutputs:

    cfg = VisionProcessingConfig(drone_dim=drone_dim, dt=dt)

    left = _process_view(
        gt_centroids_px=ground_truth.centroids_left,
        pred_centroids_px=output.pred.centroids_left,
        gt_angles_rad=ground_truth.angles_left,
        pred_angles_rad=output.pred.angles_left,
        cfg=cfg,
    )
    right = _process_view(
        gt_centroids_px=ground_truth.centroids_right,
        pred_centroids_px=output.pred.centroids_right,
        gt_angles_rad=ground_truth.angles_right,
        pred_angles_rad=output.pred.angles_right,
        cfg=cfg,
    )
    # 2) prepend gt at t=0 for predicted vision (keep length T)
    y_left_pred  = _prepend_gt0(output.pred.y_left,  ground_truth.y_left,  time_dim=1)

    if drone_dim == 3:
        y_right_pred = _prepend_gt0(output.pred.y_right, ground_truth.y_right, time_dim=1)
    else:
        y_right_pred = torch.zeros_like(y_left_pred)

    # 3) build state from processed centroids/angles
    state_gt, state_pred = _build_states(left, right, cfg)

    # 4) unscale inputs
    inputs_physical = inverse_scale_controls(u_gt, u_scaler)

    x_data = ground_truth.x_data
    angles_indexes = utils.get_angle_indexes(drone_dim)

    x_data_deg = x_data.clone()
    for i in angles_indexes:
        x_data_deg[:, :, i] = x_data[:, :, i] * RAD_TO_DEG

    # 5) build outputs
    new_pred = replace(
        output.pred,
        centroids_left=left.centroids_pred_m,
        angles_left=left.angles_pred,
        centroids_right=right.centroids_pred_m,
        angles_right=right.angles_pred,
        state=state_pred,
        y_left=y_left_pred,
        y_right=y_right_pred,
    )
    new_gt = replace(
        ground_truth,
        centroids_left=left.centroids_gt_m,
        angles_left=left.angles_gt,
        centroids_right=right.centroids_gt_m,
        angles_right=right.angles_gt,
        state=state_gt,
        x_data=x_data_deg,
    )
    state_processed = _process_state_xgt(x_gt, cfg.drone_dim)

    return VisionValForwardOutputs(
        rec=output.rec,
        pred=new_pred,
        g_t=new_gt,
        inputs_scaled=u_gt,
        state=state_processed,
        inputs_physical=inputs_physical,
    )

# =============================================================================
# View-wise feature processing
# =============================================================================

@dataclass(frozen=True)
class _ViewProcessed:
    centroids_gt_m: Tensor        # (B,T,N,2)
    centroids_pred_m: Tensor      # (B,T,N,2) with gt0 prepended
    angles_gt: Tensor             # (B,T,N) in deg or rad depending cfg
    angles_pred: Tensor           # (B,T,N) with gt0 prepended, in deg or rad


def _process_view(
        gt_centroids_px: Tensor,
        pred_centroids_px: Tensor,
        gt_angles_rad: Tensor,
        pred_angles_rad: Tensor,
        cfg: VisionProcessingConfig,
) -> _ViewProcessed:
    pred_centroids_full_px = _prepend_gt0(pred_centroids_px, gt_centroids_px, time_dim=1)
    pred_angles_full_rad = _prepend_gt0(pred_angles_rad, gt_angles_rad, time_dim=1)

    gt_centroids_m = pixels_to_meters(
        gt_centroids_px,
        cfg.img_size,
        cfg.world_size
    )
    pred_centroids_full_m = pixels_to_meters(
        pred_centroids_full_px,
        cfg.img_size,
        cfg.world_size
    )

    if cfg.output_angles_in_deg:
        gt_angles = gt_angles_rad * RAD_TO_DEG
        pred_angles = pred_angles_full_rad * RAD_TO_DEG
    else:
        gt_angles = gt_angles_rad
        pred_angles = pred_angles_full_rad

    if cfg.flip_angle_sign:
        gt_angles = -gt_angles
        pred_angles = -pred_angles

    return _ViewProcessed(
        centroids_gt_m=gt_centroids_m,
        centroids_pred_m=pred_centroids_full_m,
        angles_gt=gt_angles,
        angles_pred=pred_angles,
    )


# =============================================================================
# State building (2D / 3D)
# =============================================================================

def _build_states(
        left: _ViewProcessed,
        right: _ViewProcessed,
        cfg: VisionProcessingConfig
) -> Tuple[Tensor, Tensor]:

    if cfg.drone_dim == 2:
        state_gt = _centroids_angles_to_state_2d(
            left.centroids_gt_m,
            left.angles_gt,
            cfg
        )
        state_pred = _centroids_angles_to_state_2d(
            left.centroids_pred_m,
            left.angles_pred,
            cfg
        )
        return state_gt, state_pred

    if cfg.drone_dim == 3:
        state_gt = _centroids_angles_to_state_3d(
            left.centroids_gt_m,
            left.angles_gt,
            right.centroids_gt_m,
            right.angles_gt,
            cfg,
        )
        state_pred = _centroids_angles_to_state_3d(
            left.centroids_pred_m,
            left.angles_pred,
            right.centroids_pred_m,
            right.angles_pred,
            cfg,
        )
        return state_gt, state_pred

    raise ValueError(f"Invalid drone dimension {cfg.drone_dim}")


def _centroids_angles_to_state_2d(
        centroids: Tensor,
        angles: Tensor,
        cfg: VisionProcessingConfig
) -> Tensor:

    c = centroids[:, :, cfg.point_idx, :]                 # (B,T,2)
    th = angles[:, :, cfg.point_idx].unsqueeze(-1)        # (B,T,1)

    y = c[:, :, 0:1]
    z = c[:, :, 1:2]
    theta = th

    y_dot = _finite_diff_forward(y, cfg.dt, pad=cfg.pad_mode)
    z_dot = _finite_diff_forward(z, cfg.dt, pad=cfg.pad_mode)

    theta_dot = _angle_velocity(theta, cfg.dt, pad=cfg.pad_mode)
    return torch.cat([y, z, theta, y_dot, z_dot, theta_dot], dim=-1)


def _centroids_angles_to_state_3d(
        centroids_left: Tensor,
        angles_left: Tensor,
        centroids_right: Tensor,
        angles_right: Tensor,
        cfg: VisionProcessingConfig,
) -> Tensor:
    c_left = centroids_left[:, :, 0, :]
    c_right = centroids_right[:, :, 0, :]

    phi = angles_left[:, :, 0].unsqueeze(-1)
    theta = (-1)*angles_right[:, :, 0].unsqueeze(-1)
    psi = torch.zeros_like(theta)

    x = c_right[:, :,  0:1]
    y = c_left[:, :, 0:1]
    z = c_left[:, :, 1:2]

    x_dot = _finite_diff_forward(x, cfg.dt, pad=cfg.pad_mode)
    y_dot = _finite_diff_forward(y, cfg.dt, pad=cfg.pad_mode)
    z_dot = _finite_diff_forward(z, cfg.dt, pad=cfg.pad_mode)

    phi_dot = _angle_velocity(phi, cfg.dt, pad=cfg.pad_mode)
    theta_dot = _angle_velocity(theta, cfg.dt, pad=cfg.pad_mode)
    psi_dot = torch.zeros_like(phi_dot)

    state = [x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot]
    return torch.cat(state, dim=-1)


# =============================================================================
# Numerics / utilities
# =============================================================================
def pixels_to_meters(centroids_px: Tensor, img_size: int, world_size: float) -> Tensor:
    scale = world_size / img_size
    x_px = centroids_px[..., 0]
    y_px = centroids_px[..., 1]
    x_centered = x_px - img_size / 2
    y_centered = y_px - img_size / 2
    x_m = x_centered * scale
    y_m = y_centered * scale
    return torch.stack((x_m, -y_m), dim=-1)


def _prepend_gt0(pred: Tensor, gt: Tensor, time_dim: int = 1) -> Tensor:
    gt0 = gt.select(time_dim, 0).unsqueeze(time_dim)
    return torch.cat([gt0, pred], dim=time_dim)


def inverse_scale_controls(u_scaled: Tensor, scaler: StandardScaler) -> Tensor:
    u_np = u_scaled.detach().cpu().numpy()
    B, T, D = u_np.shape
    u_np = scaler.inverse_transform(u_np.reshape(-1, D))
    return torch.from_numpy(u_np.reshape(B, T, D)).to(device=u_scaled.device, dtype=u_scaled.dtype)


def _finite_diff_forward(v: Tensor, dt: float, pad: PadMode) -> Tensor:
    dv = (v[:, 1:] - v[:, :-1]) / dt
    if pad == "zeros":
        first = torch.zeros_like(v[:, :1])
    elif pad == "replicate":
        first = dv[:, :1]
    else:
        raise ValueError(pad)
    return torch.cat([first, first, dv[:, 1:]], dim=1)


def _wrap_angle_diff(theta_next: Tensor, theta_prev: Tensor, in_deg: bool) -> Tensor:
    if in_deg:
        d = theta_next - theta_prev
        return (d + 180.0) % 360.0 - 180.0
    else:
        d = theta_next - theta_prev
        return (d + np.pi) % (2 * np.pi) - np.pi


def _angle_velocity(theta: Tensor, dt: float, pad: PadMode) -> Tensor:
    # theta: (B,T,1)
    in_deg = True  # because cfg.output_angles_in_deg -> but if you support rad, make it param
    dtheta = _wrap_angle_diff(theta[:, 1:], theta[:, :-1], in_deg=in_deg) / dt
    if pad == "zeros":
        first = torch.zeros_like(dtheta[:, :1])
    elif pad == "replicate":
        first = dtheta[:, :1]
    else:
        raise ValueError(pad)
    return torch.cat([first, first, dtheta[:, 1:]], dim=1)





