from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from KoNAMIC.core.rendering import render_trajectory_sequence, create_snapshots_figure
from KoNAMIC.core.utils import to_numpy
from KoNAMIC.core.models import VisionValForwardOutputs

from .utils.extract_images import (
    extract_open_loop_pred_gt_imgs_2d,
    extract_open_loop_pred_gt_imgs_3d
)
from .single_visualizer import SinglePlotExtractors, OpenLoopSingleVisualizer
from .additional_traj_visualizer import AdditionalTrajVisualizer

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RenderOpenLoopConfig:
    modality: str                 # "vision" | "sensor"
    drone_dim: int                # 2 | 3
    dt: float
    phase: str
    epoch: int
    num_rollouts: int

    # plotting / layout
    num_columns_x: int
    num_columns_u: int
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
        save_simulation_output(output_k, rollout_dir / "results", extractors)

        # 1) State/input plots
        visualizer = OpenLoopSingleVisualizer(
            drone_dim=config.drone_dim,
            dt=config.dt,
            num_columns_states=config.num_columns_x,
            num_columns_inputs=config.num_columns_u,
            only_position=config.only_position,
            path=rollout_dir,
            extractors=extractors,
        )
        visualizer.pipeline(output_k)

        if config.modality == "vision":
            visualizer_additional = AdditionalTrajVisualizer(
                drone_dim=config.drone_dim,
                dt=config.dt,
                path=rollout_dir,
                x_gt=to_numpy(output_k.g_t.x_data),
                only_position=config.only_position
            )
            visualizer_additional.pipeline(output_k)

            if config.render_images:
                _render_rollout_images(
                    output_k,
                    rollout_dir=rollout_dir,
                    drone_dim=config.drone_dim,
                    label_text=config.label,
                )

                if config.snapshots:
                    left_dir = rollout_dir / "vision" / "left"
                    out_path = rollout_dir / "snapshots.png"
                    create_snapshots_figure(
                        im_dir=left_dir,
                        out_path=out_path,
                        step_count=config.num_steps,
                        label=config.label,
                        n_rows=config.snapshots_n_rows,
                    )

# -----------------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------------
def _render_rollout_images(
    output_k: VisionValForwardOutputs,
    rollout_dir: Path,
    drone_dim: int,
    label_text: str,
) -> None:
    if drone_dim == 2:
        _render_rollout_images_2d(output_k, rollout_dir, label_text)
    elif drone_dim == 3:
        _render_rollout_images_3d(output_k, rollout_dir, label_text)
    else:
        raise ValueError(f"Invalid drone dimension: {drone_dim}")


def save_simulation_output(
    output: Any,
    path: Path,
    extractors: SinglePlotExtractors
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "x_gt": extractors.get_x_gt(output),
        "x_pred": extractors.get_x_pred(output),
        "u": extractors.get_u(output),
    }
    np.savez_compressed(path.with_suffix(".npz"), **data)


def _render_rollout_images_2d(
    output_k: VisionValForwardOutputs,
    rollout_dir: Path,
    label_text: str,
) -> None:
    pred_imgs, gt_imgs = extract_open_loop_pred_gt_imgs_2d(output_k)
    out_dir_left = rollout_dir / "vision" / "left"
    out_dir_left.mkdir(parents=True, exist_ok=True)

    render_trajectory_sequence(
        pred_imgs=pred_imgs,
        gt_imgs=gt_imgs,
        path_results=out_dir_left,
        label_text=label_text,
    )


def _render_rollout_images_3d(
    output_k: VisionValForwardOutputs,
    rollout_dir: Path,
    label_text: str,
) -> None:
    pred_left, gt_left, pred_right, gt_right = extract_open_loop_pred_gt_imgs_3d(output_k)
    out_dir_left = rollout_dir / "vision" / "left"
    out_dir_left.mkdir(parents=True, exist_ok=True)
    render_trajectory_sequence(
        pred_imgs=pred_left,
        gt_imgs=gt_left,
        path_results=out_dir_left,
        label_text=label_text
    )
    out_dir_right = rollout_dir / "vision" / "right"
    out_dir_right.mkdir(parents=True, exist_ok=True)
    render_trajectory_sequence(
        pred_imgs=pred_right,
        gt_imgs=gt_right,
        path_results=out_dir_right,
        label_text=label_text
    )


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