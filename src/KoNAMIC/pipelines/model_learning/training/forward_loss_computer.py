from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeAlias

import torch
from torch import Tensor

from KoNAMIC import config
from KoNAMIC.core.models import SensorKoopModel, SensorValForwardOutputs, VisionKoopModel
from KoNAMIC.core.models.outputs.vision_outputs import GroundTruth
from KoNAMIC.core.rendering.features.geometric_features_diff import (
    compute_angles_diff,
    compute_centroids_diff,
)

BatchType: TypeAlias = Sequence[Tensor]
EffectiveWeightFn: TypeAlias = Callable[[float, str], float]
PhasesActiveFn: TypeAlias = Callable[[], list[bool]]


class SensorForwardLossComputer:
    """
    Runs the sensor model forward pass and computes the associated loss.

    This class intentionally does not call optimizer.zero_grad(), backward(),
    gradient clipping, or optimizer.step().
    """

    def __init__(
        self,
        koop_model: SensorKoopModel,
        loss_computer,
    ) -> None:
        self.koop_model = koop_model
        self.loss_computer = loss_computer
        self.device = next(self.koop_model.parameters()).device

    def compute(self, batch: BatchType, num_steps: int):
        x_gt_scaled, u_traj_scaled = (
            x.to(self.device, non_blocking=True) for x in batch
        )

        rec, pred = self.koop_model.forward(
            x_gt_scaled[:, 0],
            u_traj_scaled,
            num_steps,
        )

        z_proj = self.koop_model.batch_projection(x_gt_scaled)

        model_outputs = SensorValForwardOutputs(
            rec=rec,
            pred=pred,
            proj=z_proj,
            state_gt_scaled=x_gt_scaled,
            inputs_scaled=u_traj_scaled,
        )

        return self.loss_computer.compute(model_outputs, self.device)


class VisionForwardLossComputer:
    """
    Runs the vision model forward pass and computes the associated loss.
    """

    def __init__(
        self,
        koop_model: VisionKoopModel,
        loss_computer,
        *,
        phases_active: PhasesActiveFn | None = None,
        effective_weight: EffectiveWeightFn | None = None,
    ) -> None:
        self.koop_model = koop_model
        self.loss_computer = loss_computer
        self.device = next(self.koop_model.parameters()).device
        self.phases_active = phases_active or (lambda: [False])
        self.effective_weight = effective_weight or (lambda base, _key: base)

    def compute(self, batch: BatchType, num_steps: int):
        y_traj, u_traj, x_data = (
            x.to(self.device, non_blocking=True) for x in batch
        )

        y_traj = y_traj[:, :num_steps]
        u_traj = u_traj[:, :num_steps]
        x_data = x_data[:, :num_steps]

        model_outputs = self.koop_model.forward(
            y_init=y_traj[:, 0],
            u_traj=u_traj,
            num_steps=num_steps,
        )
        z_proj = self.koop_model.batch_projection(y_traj, u_traj)
        targets = _build_vision_ground_truth(
            y_traj=y_traj,
            x_data=x_data,
            num_views=self.koop_model.num_views,
        )

        return self.loss_computer.compute(
            model_outputs=model_outputs,
            z_proj=z_proj,
            targets=targets,
            phases_active=self.phases_active(),
            effective_weight=self.effective_weight,
            num_views=self.koop_model.num_views,
        )


def build_forward_loss_computer(
    *,
    modality: config.Modality,
    koop_model: SensorKoopModel | VisionKoopModel,
    loss_computer,
    phases_active: PhasesActiveFn | None = None,
    effective_weight: EffectiveWeightFn | None = None,
) -> SensorForwardLossComputer | VisionForwardLossComputer:
    if modality is config.Modality.SENSOR:
        if not isinstance(koop_model, SensorKoopModel):
            raise TypeError("Sensor modality expects SensorKoopModel.")
        return SensorForwardLossComputer(
            koop_model=koop_model,
            loss_computer=loss_computer,
        )

    if modality is config.Modality.VISION:
        if not isinstance(koop_model, VisionKoopModel):
            raise TypeError("Vision modality expects VisionKoopModel.")
        return VisionForwardLossComputer(
            koop_model=koop_model,
            loss_computer=loss_computer,
            phases_active=phases_active,
            effective_weight=effective_weight,
        )

    raise ValueError(f"Unknown modality: {modality.key}")


def _build_vision_ground_truth(
    *,
    y_traj: Tensor,
    x_data: Tensor,
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
        centroids_left=compute_centroids_diff(y_left),
        angles_left=compute_angles_diff(y_left),
        centroids_right=compute_centroids_diff(y_right),
        angles_right=compute_angles_diff(y_right),
        x_data=x_data,
    )


def _extract_current_views(y_traj: Tensor, num_views: int) -> Tensor:
    if y_traj.dim() != 5:
        raise ValueError(f"Expected y_traj as (B,T,C,H,W), got {tuple(y_traj.shape)}")
    if y_traj.shape[2] < num_views:
        raise ValueError(
            f"Expected at least {num_views} image channels, got {y_traj.shape[2]}."
        )
    return y_traj[:, :, -num_views:]
