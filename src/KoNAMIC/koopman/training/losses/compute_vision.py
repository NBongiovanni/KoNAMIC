from typing import TypedDict

import torch
import torch.nn as nn

from KoNAMIC.koopman.models.outputs.vision_outputs import ForwardOutputs, GroundTruth


class VisionSubLosses(TypedDict):
    y_pred: float
    y_rec: float
    z: float
    c: float
    horizontal: float
    vertical: float
    angle: float
    iou: float


VisionFullLoss = tuple[torch.Tensor, VisionSubLosses]


class VisionLossComputer:
    def __init__(self, base):
        self.base = base
        self.mse = nn.MSELoss()
        self.bce_logits = nn.BCEWithLogitsLoss()

    def compute(
        self,
        model_outputs: ForwardOutputs,
        z_proj: torch.Tensor,
        targets: GroundTruth,
        phases_active: list[bool],
        effective_weight,
        num_views: int,
    ) -> VisionFullLoss:
        """
        model_outputs.pred.* has horizon T-1.
        targets.* and z_proj have horizon T.

        Therefore, predictions are compared with targets[:, 1:num_steps].
        """
        pred = model_outputs.pred
        rec = model_outputs.rec

        num_pred_steps = pred.z.shape[1]
        target_slice = slice(1, num_pred_steps + 1)

        # --- image losses ---
        loss_y_rec_left = self.bce_logits(
            rec.y_logits_left,
            targets.y_left[:, 0],
        )
        loss_y_pred_left = self.bce_logits(
            pred.y_logits_left,
            targets.y_left[:, target_slice],
        )

        if num_views == 2:
            loss_y_rec_right = self.bce_logits(
                rec.y_logits_right,
                targets.y_right[:, 0],
            )
            loss_y_pred_right = self.bce_logits(
                pred.y_logits_right,
                targets.y_right[:, target_slice],
            )
            loss_y_rec = 0.5 * (loss_y_rec_left + loss_y_rec_right)
            loss_y_pred = 0.5 * (loss_y_pred_left + loss_y_pred_right)
        elif num_views == 1:
            loss_y_rec = loss_y_rec_left
            loss_y_pred = loss_y_pred_left
        else:
            raise ValueError(f"Invalid num_views={num_views}")

        # --- latent prediction loss ---
        loss_z = self.mse(
            pred.z,
            z_proj[:, target_slice],
        )

        # --- geometric losses, left view ---
        loss_c_left = self.mse(
            pred.centroids_left,
            targets.centroids_left[:, target_slice],
        )
        loss_angle_left = self.mse(
            pred.angles_left,
            targets.angles_left[:, target_slice],
        )

        if num_views == 2:
            loss_c_right = self.mse(
                pred.centroids_right,
                targets.centroids_right[:, target_slice],
            )
            loss_angle_right = self.mse(
                pred.angles_right,
                targets.angles_right[:, target_slice],
            )
            loss_c = 0.5 * (loss_c_left + loss_c_right)
            loss_angle = 0.5 * (loss_angle_left + loss_angle_right)
        else:
            loss_c = loss_c_left
            loss_angle = loss_angle_left

        horizontal = self.mse(
            pred.centroids_left[..., 0],
            targets.centroids_left[:, target_slice, ..., 0],
        )

        vertical = self.mse(
            pred.centroids_left[..., 1],
            targets.centroids_left[:, target_slice, ..., 1],
        )

        iou = _soft_iou(
            pred.y_left,
            targets.y_left[:, target_slice],
        )

        full_loss = (
            self.base["y_pred"] * loss_y_pred
            + self.base["y_rec"] * loss_y_rec
            + self.base["z"] * loss_z
        )

        if phases_active[0]:
            full_loss = (
                full_loss
                + effective_weight(self.base["c"], "c") * loss_c
                + effective_weight(self.base["a"], "a") * loss_angle
            )

        sub_losses: VisionSubLosses = {
            "y_pred": float(loss_y_pred.detach().cpu().item()),
            "y_rec": float(loss_y_rec.detach().cpu().item()),
            "z": float(loss_z.detach().cpu().item()),
            "c": float(loss_c.detach().cpu().item()),
            "horizontal": float(horizontal.detach().cpu().item()),
            "vertical": float(vertical.detach().cpu().item()),
            "angle": float(loss_angle.detach().cpu().item()),
            "iou": float(iou.detach().cpu().item()),
        }
        return full_loss, sub_losses


def _soft_iou(y_pred: torch.Tensor, y_gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Soft IoU metric for binary-like images.
    y_pred and y_gt are expected in [0, 1].
    """
    intersection = (y_pred * y_gt).sum(dim=(-1, -2, -3))
    union = (y_pred + y_gt - y_pred * y_gt).sum(dim=(-1, -2, -3))
    return (intersection / (union + eps)).mean()