from __future__ import annotations

from dataclasses import replace
from typing import Any

from KoNAMIC.core.models import SensorValForwardOutputs, VisionValForwardOutputs


def extract_one_rollout_sensor(
    output: SensorValForwardOutputs,
    idx_traj: int,
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
        state_gt_physical=(
            None
            if output.state_gt_physical is None
            else _maybe_index(output.state_gt_physical, idx_traj)
        ),
        inputs_scaled=_maybe_index(output.inputs_scaled, idx_traj),
        inputs_physical=(
            None
            if output.inputs_physical is None
            else _maybe_index(output.inputs_physical, idx_traj)
        ),
    )


def extract_one_rollout_vision(
    output: VisionValForwardOutputs,
    idx_traj: int,
) -> VisionValForwardOutputs:
    rec = replace(
        output.rec,
        y_logits_right=_maybe_index(output.rec.y_logits_right, idx_traj),
        y_right=_maybe_index(output.rec.y_right, idx_traj),
        y_logits_left=_maybe_index(output.rec.y_logits_left, idx_traj),
        y_left=_maybe_index(output.rec.y_left, idx_traj),
    )
    pred = replace(
        output.pred,
        y_left=_maybe_index(output.pred.y_left, idx_traj),
        y_logits_left=_maybe_index(output.pred.y_logits_left, idx_traj),
        y_right=_maybe_index(output.pred.y_right, idx_traj),
        y_logits_right=_maybe_index(output.pred.y_logits_right, idx_traj),
        z=_maybe_index(output.pred.z, idx_traj),
        centroids_right=_maybe_index(output.pred.centroids_right, idx_traj),
        angles_right=_maybe_index(output.pred.angles_right, idx_traj),
        centroids_left=_maybe_index(output.pred.centroids_left, idx_traj),
        angles_left=_maybe_index(output.pred.angles_left, idx_traj),
        state=(
            None
            if output.pred.state is None
            else _maybe_index(output.pred.state, idx_traj)
        ),
        state_right=(
            None
            if output.pred.state_right is None
            else _maybe_index(output.pred.state_right, idx_traj)
        ),
    )
    g_t = replace(
        output.g_t,
        y_left=_maybe_index(output.g_t.y_left, idx_traj),
        y_right=_maybe_index(output.g_t.y_right, idx_traj),
        centroids_left=_maybe_index(output.g_t.centroids_left, idx_traj),
        angles_left=_maybe_index(output.g_t.angles_left, idx_traj),
        centroids_right=_maybe_index(output.g_t.centroids_right, idx_traj),
        angles_right=_maybe_index(output.g_t.angles_right, idx_traj),
        state=(
            None
            if output.g_t.state is None
            else _maybe_index(output.g_t.state, idx_traj)
        ),
        state_right=(
            None
            if output.g_t.state_right is None
            else _maybe_index(output.g_t.state_right, idx_traj)
        ),
        x_data=(
            None
            if output.g_t.x_data is None
            else _maybe_index(output.g_t.x_data, idx_traj)
        ),
    )

    return replace(
        output,
        rec=rec,
        pred=pred,
        g_t=g_t,
        inputs_scaled=_maybe_index(output.inputs_scaled, idx_traj),
        state=_maybe_index(output.state, idx_traj),
        inputs_physical=(
            None
            if output.inputs_physical is None
            else _maybe_index(output.inputs_physical, idx_traj)
        ),
    )


def get_rollout_extractor_for_modality(modality: str):
    if modality == "sensor":
        return extract_one_rollout_sensor
    if modality == "vision":
        return extract_one_rollout_vision

    raise ValueError(f"Unsupported modality: {modality}")


def _maybe_index(x: Any, idx: int) -> Any:
    try:
        return x[idx]
    except Exception:
        return x
