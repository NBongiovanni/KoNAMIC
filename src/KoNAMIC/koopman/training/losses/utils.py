from typing import TypeAlias

import math

import numpy as np
import torch

SubLosses: TypeAlias = dict[str, float]


def mean_sub_losses(items: list[dict]) -> dict:
    if not items:
        return {}

    result = {}
    keys = items[0].keys()

    for key in keys:
        values = [item[key] for item in items]

        if isinstance(values[0], dict):
            result[key] = mean_sub_losses(values)
        else:
            result[key] = float(np.mean(values, dtype=np.float64))

    return result


def reduce_sub_losses(sub_losses_arr: list[SubLosses]) -> SubLosses:
    return mean_sub_losses(sub_losses_arr)


def build_state_rmse_scale(
    state_names: list[str],
    device: torch.device,
) -> torch.Tensor:
    scale = torch.ones(len(state_names), device=device)

    angular_names = {
        "phi",
        "theta",
        "psi",
        "phi_dot",
        "theta_dot",
        "psi_dot",
    }

    for i, name in enumerate(state_names):
        if name in angular_names:
            scale[i] = 180.0 / math.pi

    return scale
