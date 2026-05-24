from __future__ import annotations

import torch
from torch import Tensor

from KoNAMIC.pipelines.data_pipeline import compute_angles_diff, compute_centroids_diff
from KoNAMIC.core.models import GroundTruth


def get_channel_slices(num_views: int, delay: int) -> dict[str, slice]:
    """
    Channel layout convention:
      - delay=1 -> concat([views@t-1, views@t])      => C = 2*num_views
      - delay=2 -> concat([views@t-2, views@t-1, views@t]) => C = 3*num_views

    Returns a dict mapping:
      view{v}_km{d} for d in {delay,...,1} and view{v}_k (always).
    Each slice selects exactly 1 channel.
    """
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}")

    slices: dict[str, slice] = {}

    # Offsets for temporal blocks in the concatenated channel dimension
    # block 0 = t-delay, ..., block delay-1 = t-1, block delay = t
    for v in range(num_views):
        # past blocks (if any)
        for d in range(delay, 0, -1):
            block_index = delay - d  # d=delay -> 0, d=1 -> delay-1
            ch = v + block_index * num_views
            slices[f"view{v}_km{d}"] = slice(ch, ch + 1)

        # current block (t)
        ch_k = v + delay * num_views
        slices[f"view{v}_k"] = slice(ch_k, ch_k + 1)
    return slices


def build_ground_truth_from_images(
        y_data: Tensor,
        x_data: Tensor,
        drone_dim: int,
        delay: int = 2,
) -> GroundTruth:
    # Config
    num_views = 2 if drone_dim == 3 else 1
    idx = get_channel_slices(num_views, delay)

    def _geom_from_slice(ch_slice):
        y = y_data[:, :, ch_slice]
        centroids = compute_centroids_diff(y)
        angles = compute_angles_diff(y)
        return y, centroids, angles

    # Left (always present)
    y_left, centroids_left, angles_left = _geom_from_slice(idx["view0_k"])

    # Right (depends on drone_dim)
    if drone_dim == 3:
        y_right, centroids_right, angles_right = _geom_from_slice(idx["view1_k"])
    elif drone_dim == 2:
        y_right = torch.zeros_like(y_left)
        centroids_right = torch.zeros_like(centroids_left)
        angles_right = torch.zeros_like(angles_left)
    else:
        raise ValueError(f"Invalid drone_dim={drone_dim}. Expected 2 or 3.")

    return GroundTruth(
        y_left=y_left,
        y_right=y_right,
        centroids_left=centroids_left,
        angles_left=angles_left,
        centroids_right=centroids_right,
        angles_right=angles_right,
        x_data=x_data,
    )
