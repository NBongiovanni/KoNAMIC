from __future__ import annotations

import numpy as np

def moving_average_along_time(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x

    pad_left = window // 2
    pad_right = window - 1 - pad_left

    x_pad = np.pad(
        x,
        pad_width=((0, 0), (pad_left, pad_right), (0, 0)),
        mode="edge",
    )

    kernel = np.ones(window, dtype=float) / window
    out = np.empty_like(x)

    for i in range(x.shape[0]):
        for j in range(x.shape[2]):
            out[i, :, j] = np.convolve(x_pad[i, :, j], kernel, mode="valid")

    return out