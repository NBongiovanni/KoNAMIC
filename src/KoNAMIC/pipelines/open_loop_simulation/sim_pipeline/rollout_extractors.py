from dataclasses import replace
from typing import Any

from KoNAMIC.core.models import VisionValForwardOutputs, SensorValForwardOutputs


def extract_one_rollout_vision(
        output: VisionValForwardOutputs,
        idx_traj: int
) -> VisionValForwardOutputs:

    rec = type(output.rec)(
        y_left=output.rec.y_left[idx_traj],
        y_logits_left=output.rec.y_logits_left[idx_traj],
        y_right=output.rec.y_right[idx_traj],
        y_logits_right=output.rec.y_logits_right[idx_traj],
    )

    pred = type(output.pred)(
        y_left=output.pred.y_left[idx_traj],
        y_logits_left=output.pred.y_logits_left[idx_traj],
        y_right=output.pred.y_right[idx_traj],
        y_logits_right=output.pred.y_logits_right[idx_traj],
        z=output.pred.z[idx_traj],
        centroids_left=output.pred.centroids_left[idx_traj],
        angles_left=output.pred.angles_left[idx_traj],
        centroids_right=output.pred.centroids_right[idx_traj],
        angles_right=output.pred.angles_right[idx_traj],
        state=None if not hasattr(output.pred, "state") else output.pred.state[idx_traj],
    )

    g_t = type(output.g_t)(
        y_left=output.g_t.y_left[idx_traj],
        y_right=output.g_t.y_right[idx_traj],
        centroids_left=output.g_t.centroids_left[idx_traj],
        angles_left=output.g_t.angles_left[idx_traj],
        centroids_right=output.g_t.centroids_right[idx_traj],
        angles_right=output.g_t.angles_right[idx_traj],
        state=None if not hasattr(output.g_t, "state") else output.g_t.state[idx_traj],
        x_data=None if not hasattr(output.g_t, "x_data") else output.g_t.x_data[idx_traj],
    )

    inputs = output.inputs_scaled[idx_traj]
    inputs_physical = output.inputs_physical[idx_traj]
    x = output.state[idx_traj]

    return type(output)(
        rec=rec,
        pred=pred,
        g_t=g_t,
        inputs_scaled=inputs,
        state=x,
        inputs_physical=inputs_physical,
    )

def _maybe_index(x: Any, idx: int) -> Any:
    # indexe si possible (Tensor, list, etc.), sinon renvoie x
    try:
        return x[idx]
    except Exception:
        return x

def extract_one_rollout_sensor(
        output: SensorValForwardOutputs,
        idx_traj: int
) -> SensorValForwardOutputs:

    pred = replace(
        output.pred,
        state=_maybe_index(output.pred.state, idx_traj),
        z=_maybe_index(output.pred.z, idx_traj),
    )

    return replace(
        output,
        rec=_maybe_index(output.rec, idx_traj),
        pred=pred,
        proj=_maybe_index(output.proj, idx_traj),
        state_gt_scaled=_maybe_index(output.state_gt_scaled, idx_traj),
        state_gt_physical=None if output.state_gt_physical is None else _maybe_index(output.state_gt_physical, idx_traj),
        inputs_scaled=_maybe_index(output.inputs_scaled, idx_traj),
        inputs_physical=None if output.inputs_physical is None else _maybe_index(output.inputs_physical, idx_traj),
    )