from typing import TypedDict, cast

import numpy as np
import torch
import torch.nn as nn

from KoNAMIC.core.models.outputs.sensor_outputs import SensorValForwardOutputs

class SensorsSubLosses(TypedDict):
    rec: float
    pred_z: float
    pred_x: float
    pred_position: float

SensorFullLoss = tuple[torch.Tensor, SensorsSubLosses]


class SensorLossComputer:
    def __init__(self, base: dict):
        self.base = base
        self.mse = nn.MSELoss()

    def compute(
            self, models_outputs: SensorValForwardOutputs, device: torch.device,
) -> SensorFullLoss:
        alphas = self.base

        x_rec = models_outputs.rec
        z_proj = models_outputs.proj
        z_pred = models_outputs.pred.z
        x_pred = models_outputs.pred.state
        x_gt = models_outputs.state_gt_scaled

        loss_x_rec = self.mse(x_rec, x_gt[:, 0].to(device))
        loss_pred_x = self.mse(x_pred[:, 1:], x_gt[:, 1:].to(device))
        loss_position = self.mse(x_pred[:, 1:, :3], x_gt[:, 1:, :3].to(device))
        loss_z_pred = self.mse(z_pred[:, 1:], z_proj[:, 1:])

        full_loss = (
                alphas["rec"] * loss_x_rec +
                alphas["pred_z"] * loss_z_pred +
                alphas["pred_x"] * loss_pred_x
        )

        sub_losses: SensorsSubLosses = {
            "rec": float(loss_x_rec.detach().cpu().item()),
            "pred_z": float(loss_z_pred.detach().cpu().item()),
            "pred_x": float(loss_pred_x.detach().cpu().item()),
            "pred_position": float(loss_position.detach().cpu().item()),
        }
        return full_loss, sub_losses


def mean_sub_losses(items: list[SensorsSubLosses]) -> SensorsSubLosses:
    keys = items[0].keys()
    result = {
        k: float(np.mean([d[k] for d in items], dtype=np.float64))
        for k in keys
    }
    return cast(SensorsSubLosses, result)