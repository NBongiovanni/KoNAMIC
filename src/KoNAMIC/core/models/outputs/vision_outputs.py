from dataclasses import dataclass
import torch
from torch import Tensor
from typing import Optional

@dataclass
class Pred:
    """
    Predicted quantities over future steps (t+1..T).
    """
    y_left: Tensor
    y_logits_left: Tensor
    y_right: Tensor
    y_logits_right: Tensor
    z: Tensor
    centroids_right: Tensor
    angles_right: Tensor
    centroids_left: Tensor
    angles_left: Tensor
    state: Optional[Tensor] = None
    state_right: Optional[Tensor] = None

    def build_state_right(self) -> Tensor:
        """
        Returns state (x, y, theta) for right view.
        Shape: (batch, T, 3)
        """
        return _build_state_from_centroids_angles(
            self.centroids_right,
            self.angles_right,
        )

    def build_state_left(self) -> Tensor:
        """
        Returns state (x, y, theta) for left view.
        Shape: (batch, T, 3)
        """
        return _build_state_from_centroids_angles(
            self.centroids_left,
            self.angles_left,
        )

    @staticmethod
    def augment_with_velocity(state: Tensor, dt: float) -> Tensor:
        """
        Compute finite difference velocity and append to state.

        state: (batch, T, 3)
        returns: (batch, T-1, 6)
        """
        vel = (state[:, 1:] - state[:, :-1]) / dt
        state_cut = state[:, 1:]

        return torch.cat([state_cut, vel], dim=-1)



def _build_state_from_centroids_angles(centroids: Tensor, angles: Tensor) -> Tensor:
    if centroids.ndim == 4 and centroids.shape[-2] == 1:
        centroids = centroids.squeeze(-2)
    if centroids.shape[-1] != 2:
        raise ValueError(f"Expected centroids last dimension to be 2, got {centroids.shape}.")

    if angles.ndim == centroids.ndim - 1:
        angles = angles.unsqueeze(-1)
    if angles.ndim == 3 and angles.shape[-1] == 1:
        pass
    elif angles.ndim == 4 and angles.shape[-2] == 1 and angles.shape[-1] == 1:
        angles = angles.squeeze(-2)
    else:
        raise ValueError(f"Unexpected angles shape: {angles.shape}.")

    return torch.cat([centroids, angles], dim=-1)

@dataclass
class Rec:
    """
    Reconstruction of the initial frame.

    Attributes:
        y (Tensor): Image logits for the reconstruction of y_t, shape (B, 1, H, W)
                    or (B, 1, 1, H, W) depending on caller context.
    """
    y_logits_right: Tensor
    y_right: Tensor
    y_logits_left: Tensor
    y_left: Tensor


@dataclass
class GroundTruth:
    y_left: Tensor
    y_right: Tensor
    centroids_left: Tensor
    angles_left: Tensor
    centroids_right: Tensor
    angles_right: Tensor
    state: Optional[Tensor] = None
    state_right: Optional[Tensor] = None
    x_data: Optional[Tensor] = None

@dataclass
class ForwardOutputs:
    """
    Container for all outputs of a forward pass used during training.

    Attributes:
        rec (Rec): Reconstruction of the initial frame.
        pred (Pred): Predictions for t+1..T.
    """
    rec: Rec
    pred: Pred


@dataclass
class VisionValForwardOutputs:
    """
    Extended container used for validation/visualization.

    Attributes:
        proj, rec, pred, g_t: Same as in `ForwardOutputs`.
        inputs_scaled (Tensor): Original inputs as fed to the model/dataloader.
        state (Tensor): Additional state/pose info per sample if available.
    """
    rec: Rec
    pred: Pred
    g_t: GroundTruth
    inputs_scaled: Tensor
    state: Tensor
    inputs_physical: Optional[Tensor] = None