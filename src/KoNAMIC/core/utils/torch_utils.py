import torch
import numpy as np


def to_numpy(x):
    """
    Convert torch.Tensor or np.ndarray to np.ndarray.
    Keeps np.ndarray as-is.
    """
    if isinstance(x, np.ndarray):
        return x
    # torch tensor
    return x.detach().cpu().squeeze(0).numpy()


def load_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Torch device: {}".format(device))
    return device