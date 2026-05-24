from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import numpy as np

from .config import Dataset


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


def post_process(dataset: Dataset, cfg: DataGenerationConfig) -> Dataset:
    idx = np.arange(0, dataset.states.shape[1], cfg.ds_step)

    if cfg.ds_step == 1:
        return dataset

    states = moving_average_along_time(dataset.states, cfg.smooth_window)[:, idx, :]
    inputs = moving_average_along_time(dataset.inputs, cfg.smooth_window)[:, idx, :]
    states_ref = moving_average_along_time(dataset.states_ref, cfg.smooth_window)[:, idx, :]
    time = dataset.time[idx]

    return Dataset(
        states=states,
        inputs=inputs,
        states_ref=states_ref,
        time=time,
    )
