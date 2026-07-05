from torch import nn

def make_gn(
        num_channels: int,
        max_groups: int,
        min_channels_per_group: int,
        eps: float
) -> nn.GroupNorm:

    C = num_channels
    num_groups = min(max_groups, max(1, C // min_channels_per_group))
    while num_groups > 1 and (C % num_groups) != 0:
        num_groups -= 1
    return nn.GroupNorm(
        num_groups=num_groups,
        num_channels=C,
        eps=eps,
        affine=True,
    )