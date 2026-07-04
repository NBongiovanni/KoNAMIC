from typing import TypedDict, Sequence

import torch
import torch.nn as nn

from KoNAMIC.core.models.outputs.sensor_outputs import SensorValForwardOutputs
from .utils import build_state_rmse_scale


class SensorsSubLosses(TypedDict):
    y_rec: float
    y_pred: float
    z_pred: float
    state_rmse: dict[str, float]


SensorFullLoss = tuple[torch.Tensor, SensorsSubLosses]


class SensorLossComputer:
    def __init__(
        self,
        base,
        state_names: list[str],
        state_rmse_units_scale: Sequence[float] | None = None,
    ):
        self.base = base
        self.state_names = state_names
        self.state_rmse_units_scale = state_rmse_units_scale
        self.mse = nn.MSELoss()

        if state_rmse_units_scale is not None and len(state_rmse_units_scale) != len(state_names):
            raise ValueError(
                "state_rmse_units_scale length must match state_names length: "
                f"got {len(state_rmse_units_scale)} and {len(state_names)}."
            )

    def compute(
            self,
            models_outputs: SensorValForwardOutputs,
            device: torch.device,
    ) -> SensorFullLoss:
        alphas = self.base

        x_rec = models_outputs.rec
        z_proj = models_outputs.proj
        z_pred = models_outputs.pred.z
        x_pred = models_outputs.pred.state
        x_gt = models_outputs.state_gt_scaled.to(device)

        loss_x_rec = self.mse(x_rec, x_gt[:, 0])
        loss_pred_x = self.mse(x_pred[:, 1:], x_gt[:, 1:])
        loss_z_pred = self.mse(z_pred[:, 1:], z_proj[:, 1:])

        full_loss = (
                alphas.y_rec * loss_x_rec
                + alphas.z_pred * loss_z_pred
                + alphas.y_pred * loss_pred_x
        )

        state_error = x_pred[:, 1:] - x_gt[:, 1:]
        if self.state_rmse_units_scale is not None:
            units_scale = torch.as_tensor(
                self.state_rmse_units_scale,
                device=state_error.device,
                dtype=state_error.dtype,
            ).view(1, 1, -1)
            state_error = state_error * units_scale

        state_rmse_tensor = torch.sqrt(
            torch.mean(
                state_error ** 2,
                dim=(0, 1),
            )
        )

        scale = build_state_rmse_scale(
            state_names=self.state_names,
            device=state_rmse_tensor.device,
        )

        state_rmse_tensor = state_rmse_tensor * scale

        state_rmse = {
            name: float(value)
            for name, value in zip(
                self.state_names,
                state_rmse_tensor.detach().cpu(),
            )
        }

        sub_losses: SensorsSubLosses = {
            "y_rec": float(loss_x_rec.detach().cpu().item()),
            "y_pred": float(loss_pred_x.detach().cpu().item()),
            "z_pred": float(loss_z_pred.detach().cpu().item()),
            "state_rmse": state_rmse,
        }
        return full_loss, sub_losses
