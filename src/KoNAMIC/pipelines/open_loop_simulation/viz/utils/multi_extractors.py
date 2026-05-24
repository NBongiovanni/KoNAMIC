from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from KoNAMIC.core.utils import to_numpy
import numpy as np

RunX = tuple[np.ndarray, np.ndarray, np.ndarray]
RunU = np.ndarray
PreparedRunX = tuple[np.ndarray, np.ndarray, np.ndarray, str]
PreparedRunU = tuple[np.ndarray, np.ndarray, str]

GetRunX = Callable[[Any, float], RunX]
GetRunU = Callable[[Any], RunU]

@dataclass(frozen=True)
class MultiPlotExtractors:
    """
    Defines how to extract/aligne arrays to plot from one rollout output.
    """
    get_run_x: GetRunX
    get_u: GetRunU

def make_multi_extractors(modality: str) -> MultiPlotExtractors:
    if modality == "vision":
        return MultiPlotExtractors(
            get_run_x=_get_run_x_vision,
            get_u=_get_u_generic,
        )
    elif modality == "sensor":
        return MultiPlotExtractors(
            get_run_x=_get_run_x_sensor,
            get_u=_get_u_generic,
        )
    else:
        raise ValueError(f"Unknown modality: {modality}")


def _get_u_generic(out) -> np.ndarray:
    if isinstance(out, dict):
        return to_numpy(out["u"])
    return to_numpy(out.inputs_physical)


def _get_run_x_vision(out, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(out, dict):
        x_gt = to_numpy(out["x_gt"])
        x_pred = to_numpy(out["x_pred"])
    else:
        x_gt = to_numpy(out.g_t.state)
        x_pred = to_numpy(out.pred.state)

    L = min(x_gt.shape[0], x_pred.shape[0])
    t = np.arange(1, L + 1) * dt
    return t, x_gt[:L], x_pred[:L]


def _get_run_x_sensor(out, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_gt = to_numpy(out["x_gt"])
    x_pred = to_numpy(out["x_pred"])

    L = min(x_gt.shape[0] - 1, x_pred.shape[0])
    t = np.arange(1, L + 1) * dt
    return t, x_gt[1:1 + L], x_pred[:L]