from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from KoNAMIC.core.systems import SystemSpec
from KoNAMIC.utils import to_numpy
from KoNAMIC.koopman.models import VisionValForwardOutputs
from KoNAMIC.pipelines.experiment_comparison import TrajectoryComparisonResult
from KoNAMIC.pipelines.open_loop_simulation.trajectories import (
    sensor_output_to_comparison_result,
    vision_output_to_comparison_result,
)

from .single_visualizer import SinglePlotExtractors, OpenLoopSingleVisualizer

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RenderOpenLoopConfig:
    modality: str                 # "vision" | "sensor"
    system_spec: SystemSpec
    dt: float
    phase: str
    epoch: int
    num_rollouts: int

    # plotting / layout
    num_columns_states: int
    num_columns_inputs: int
    only_position: bool = False

    # vision
    render_images: bool = True
    num_steps: int = 10           # used for snapshots
    snapshots: bool = True
    snapshots_n_rows: int = 2
    label: str = "Model"

    # directories
    eval_dir: Optional[Path] = None   # can be set if you want; else passed to function

# -----------------------------------------------------------------------------
# Public entrypoint
# -----------------------------------------------------------------------------
def render_open_loop_rollouts(
    config: RenderOpenLoopConfig,
    output: VisionValForwardOutputs,
    eval_dir: Path,
    extract_one_rollout,
) -> None:
    """
    Render open-loop rollouts with plots (and optionally vision + snapshots).

    Args:
        config: RenderOpenLoopConfig containing rendering parameters.
        output: full output container (all rollouts).
        eval_dir: directory where results will be written.
        extract_one_rollout: function to select rollout k from `output`.
    """
    eval_dir.mkdir(parents=True, exist_ok=True)
    extractors=_make_extractors(config.modality)

    for k in range(config.num_rollouts):
        rollout_dir=eval_dir / f"rollout_{k}"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        output_k=extract_one_rollout(output, k)
        save_simulation_output(
            output_k,
            rollout_dir / "results",
            modality=config.modality,
            dt=config.dt,
        )

        # 1) State/input plots
        visualizer = OpenLoopSingleVisualizer(
            system_spec=config.system_spec,
            dt=config.dt,
            num_columns_states=config.num_columns_states,
            num_columns_inputs=config.num_columns_inputs,
            only_position=config.only_position,
            path=rollout_dir,
            extractors=extractors,
        )
        visualizer.pipeline(output_k)

# -----------------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------------
def save_simulation_output(
    output: Any,
    path: Path,
    *,
    modality: str,
    dt: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    result = _to_comparison_result(output, modality=modality, dt=dt)
    data = {
        "time": result.time,
        "reference_state": result.reference_state,
        "compared_state": result.compared_state,
        "input_time": result.input_time,
        "inputs": result.inputs,
        "dt": np.asarray(result.dt, dtype=float),
    }
    np.savez_compressed(path.with_suffix(".npz"), **data)


def _to_comparison_result(
    output: Any,
    *,
    modality: str,
    dt: float,
) -> TrajectoryComparisonResult:
    if modality == "sensor":
        return sensor_output_to_comparison_result(output, dt=dt)
    if modality == "vision":
        return vision_output_to_comparison_result(output, dt=dt)
    raise ValueError(f"Unknown modality: {modality}")



def _make_extractors(modality: str) -> SinglePlotExtractors:
    """Select extractor functions based on modality, keeping the plotter class agnostic."""
    if modality == "vision":
        return SinglePlotExtractors(
            # get_x_gt=lambda out: to_numpy(out.g_t.x_data),
            get_x_gt=lambda out: to_numpy(out.g_t.state),
            get_x_pred=lambda out: to_numpy(out.pred.state),
            get_u=lambda out: to_numpy(out.inputs_physical),
        )

    elif modality == "sensor":
        return SinglePlotExtractors(
            get_x_gt=lambda out: to_numpy(out.state_gt_physical),
            get_x_pred=lambda out: to_numpy(out.pred.state),
            get_u=lambda out: to_numpy(out.inputs_physical),
        )
    else:
        raise ValueError(f"Unknown modality: {modality}")
