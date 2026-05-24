import torch
from torch import Tensor, nn


class MultiViewSharedDecoder(nn.Module):
    """
    Split w = [wL, wR] and decode each with the same ConvDecoder (weight sharing),
    then concatenate reconstructed vision along channel dimension.

    If each view reconstructs 1 channel (grayscale), output has 2 channels: [left_t, right_t].
    """
    def __init__(self, single_view_decoder: nn.Module, feature_shape):
        super().__init__()
        self.dec = single_view_decoder
        c_o, h_o, w_o = feature_shape
        self.w_dim_view = c_o * h_o * w_o

    def forward(self, w: Tensor) -> Tensor:
        wL = w[:, : self.w_dim_view]
        wR = w[:, self.w_dim_view :]

        yL = self.dec(wL)  # (B, 1, H, W) if out_channels=1
        yR = self.dec(wR)

        return torch.cat([yL, yR], dim=1)  # (B, 2, H, W)