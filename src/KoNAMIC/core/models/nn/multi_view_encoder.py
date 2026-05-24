from __future__ import annotations
from typing import Any, Tuple, Sequence, List
import torch
from torch import nn, Tensor


def _normalize_split_indices(split_indices: Any) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Accepts:
      - YAML list: [[0,2,4],[1,3,5]] or [[0,2],[1,3]]
      - tuple of tuples: ((0,2,4),(1,3,5)) etc.
    Returns:
      (left_indices, right_indices) as tuples of ints.
    """
    if isinstance(split_indices, str):
        raise ValueError(
            "split_indices was parsed as a string. Use YAML list syntax, e.g. "
            "split_indices: [[0,2,4],[1,3,5]]"
        )

    if not isinstance(split_indices, (list, tuple)) or len(split_indices) != 2:
        raise ValueError(f"split_indices must be a list/tuple of length 2, got {split_indices}")

    left, right = split_indices

    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
        raise ValueError(f"split_indices must contain two list/tuple elements, got {split_indices}")

    left_t = tuple(int(i) for i in left)
    right_t = tuple(int(i) for i in right)

    if len(left_t) < 1 or len(right_t) < 1:
        raise ValueError(f"split_indices elements must be non-empty, got {split_indices}")

    if len(left_t) != len(right_t):
        raise ValueError(
            f"left and right must have same number of time steps/channels, got {split_indices}"
        )
    return left_t, right_t


class MultiViewSharedEncoder(nn.Module):
    """
    Apply the SAME ConvEncoder to each view (weight sharing),
    then concatenate the features.

    Expected y shape: (B, C, H, W) where C contains interleaved views/time
    Example (num_views=2):
      delay=1 -> C=4  indices left=[0,2], right=[1,3]
      delay=2 -> C=6  indices left=[0,2,4], right=[1,3,5]
    """
    def __init__(
        self,
        single_view_encoder: nn.Module,
        split_indices: Any = ((0, 2, 4), (1, 3, 5)),
    ):
        super().__init__()
        self.enc = single_view_encoder
        self.split_indices = _normalize_split_indices(split_indices)

    def forward(self, y: Tensor) -> Tensor:
        """
        y: (B, C, H, W)
        We gather left indices -> (B, T, H, W) and right indices -> (B, T, H, W),
        then flatten T into channels before encoding: (B, T, H, W) -> (B, T, H, W) as channels => (B, T, H, W)
        but Conv expects (B, C, H, W), so C=T here (since each slice is 1 channel).
        """
        left_idx, right_idx = self.split_indices  # tuples of length T

        # (B, T, H, W)
        yL = y[:, list(left_idx), :, :]
        yR = y[:, list(right_idx), :, :]

        # If your encoder expects 1-channel input only, you have two options:
        # (A) treat the T frames as channels (works if encoder input channels can be T)
        # (B) encode each frame separately then pool/concat
        #
        # Here we do (A): frames-as-channels.
        wL = self.enc(yL)  # (B, w_dim_view)
        wR = self.enc(yR)  # (B, w_dim_view)

        return torch.cat([wL, wR], dim=1)  # (B, 2*w_dim_view)