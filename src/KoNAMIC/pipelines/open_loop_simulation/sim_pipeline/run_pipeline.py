from __future__ import annotations

import argparse

from KoNAMIC.pipelines.open_loop_simulation import (
    open_loop_simulation_sensor_pipeline,
    open_loop_simulation_vision_pipeline,
)


def run_open_loop_pipeline_from_args(
    args: argparse.Namespace,
    case,
    stamp_open_loop: str,
):
    if args.modality == "sensor":
        return open_loop_simulation_sensor_pipeline(
            case=case,
            phase=args.phase,
            num_steps=args.num_steps,
            modality=args.modality,
            drone_dim=args.drone_dim,
            stamp_open_loop=stamp_open_loop,
            seed=args.seed,
        )

    if args.modality == "vision":
        if args.dataset_version is None:
            raise ValueError("dataset_version must be provided for vision modality.")

        return open_loop_simulation_vision_pipeline(
            case=case,
            phase=args.phase,
            modality=args.modality,
            num_steps=args.num_steps,
            seed=args.seed,
            stamp_open_loop=stamp_open_loop,
            dt=case.dt,
            dataset_version=args.dataset_version,
        )

    raise ValueError(f"Unsupported modality: {args.modality}")