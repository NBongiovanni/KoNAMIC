#!/usr/bin/env python
from __future__ import annotations

import multiprocessing
import argparse

import matplotlib
matplotlib.use("Agg")

from KoNAMIC.core import utils
from KoNAMIC.core.rendering import label_from_case
from KoNAMIC.pipelines.open_loop_simulation import (
    parse_args_open_loop_simulation,
    run_open_loop_pipeline_from_args,
    render_open_loop_rollouts,
    RenderOpenLoopConfig,
    extract_one_rollout_sensor,
    extract_one_rollout_vision,
)


def main() -> None:
    args = parse_args_open_loop_simulation()
    utils.set_seed(args.seed)

    case = load_case_from_args(args)
    logger = utils.setup_logging()
    stamp_open_loop = utils.make_timestamped_dir(logger)

    simulation_output = run_open_loop_pipeline_from_args(
        args=args,
        case=case,
        stamp_open_loop=stamp_open_loop,
    )

    eval_dir = simulation_output.run_dir / "eval" / "open_loop" / stamp_open_loop

    config = RenderOpenLoopConfig(
        modality=args.modality,
        drone_dim=case.drone_dim,
        dt=case.dt,
        phase=args.phase,
        epoch=case.epoch,
        num_columns_x=2,
        num_columns_u=2,
        only_position=True,
        num_rollouts=args.num_rollouts,
        render_images=False,
        snapshots=False,
        num_steps=args.num_steps,
        label=label_from_case(case),
    )

    extract_one_rollout = get_rollout_extractor_from_args(args)

    render_open_loop_rollouts(
        config=config,
        output=simulation_output.val_output,
        eval_dir=eval_dir,
        extract_one_rollout=extract_one_rollout,
    )


def get_rollout_extractor_from_args(args: argparse.Namespace):
    if args.modality == "sensor":
        return extract_one_rollout_sensor

    if args.modality == "vision":
        return extract_one_rollout_vision

    raise ValueError(f"Unsupported modality: {args.modality}")


def load_case_from_args(args: argparse.Namespace):
    cases = utils.load_cases(args.modality)
    return cases[args.caseid]


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()