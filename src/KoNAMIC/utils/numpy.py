import numpy as np
import torch


def as_array(value) -> np.ndarray:
    return np.asarray(value, dtype=float)


def as_float(value) -> float:
    """
    Accept scalar or one-element list/array from YAML.
    """
    return float(np.asarray(value, dtype=float).reshape(-1)[0])


def to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)