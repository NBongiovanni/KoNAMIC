#!/usr/bin/env python
from __future__ import annotations

import multiprocessing

import matplotlib
matplotlib.use("Agg")

from KoNAMIC import paths, utils, config
from KoNAMIC.core.systems import create_system
from KoNAMIC.pipelines.open_loop_simulation import (
    RenderOpenLoopConfig,
    get_rollout_extractor_for_modality,
    load_open_loop_run_configs,
    open_loop_simulation_sensor_pipeline,
    parse_args_open_loop_simulation,
    render_open_loop_rollouts,
)


def main() -> None:
    args = parse_args_open_loop_simulation()
    modality = config.Modality(args.modality)
    utils.set_seed(args.seed)

    logger = utils.setup_logging()
    stamp_open_loop = paths.make_timestamp(logger)

    system_spec = create_system(args.system_name)
    run_paths = paths.build_run_paths(
        modality=modality.key,
        system_name=system_spec.system_name,
        run_status=args.run_status,
        stamp_run=args.stamp_run,
    )
    dataset_paths = paths.build_dataset_paths(
        system_name=system_spec.system_name,
        dataset_stamp=args.dataset_stamp,
    )
    model_config, data_preparation_config = load_open_loop_run_configs(
        run_paths=run_paths,
        system_spec=system_spec,
    )

    simulation_output = open_loop_simulation_sensor_pipeline(
        run_paths=run_paths,
        dataset_paths=dataset_paths,
        model_config=model_config,
        data_preparation_config=data_preparation_config,
        system_spec=system_spec,
        phase=args.phase,
        num_steps=args.num_steps,
        epoch=args.epoch,
        seed=args.seed,
    )

    eval_dir = run_paths.standalone_eval_dir("open_loop", stamp_open_loop)
    render_config = RenderOpenLoopConfig(
        modality=modality.key,
        system_dim=system_spec.system_dim,
        dt=model_config.dt,
        phase=args.phase,
        epoch=args.epoch,
        num_columns_states=2,
        num_columns_inputs=2,
        only_position=args.only_position,
        num_rollouts=args.num_rollouts,
        render_images=False,
        snapshots=False,
        num_steps=args.num_steps,
        label=args.stamp_run,
    )

    render_open_loop_rollouts(
        config=render_config,
        output=simulation_output.val_output,
        eval_dir=eval_dir,
        extract_one_rollout=get_rollout_extractor_for_modality(args.modality),
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
