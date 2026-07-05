#!/usr/bin/env python
from typing import cast

import matplotlib
matplotlib.use("Agg")

from KoNAMIC import paths, utils, config
from KoNAMIC.core.systems import create_system
from KoNAMIC.koopman.models import build_model
from KoNAMIC.core.plants import build_plant
from KoNAMIC.core.scenarios import build_scenario_generator, load_scenario_gen_config
from KoNAMIC.core.scaling import DatasetScalers
from KoNAMIC.pipelines.data_preparation import (
    VisionBuilder,
    VisionPreparationConfig,
    SensorBuilder,
    prepare_vision_dataset,
)
from KoNAMIC.pipelines.model_learning import (
    Trainer,
    TrainingPipelineConfig,
    parse_learning_args,
    TrainingEvaluator,
    build_training_context,
    save_effective_run_config,
)


def main() -> None:
    """
    Entry point used to train a Koopman-based model from either sensor or visual data.

    The script performs the complete training setup:
    - parse command-line arguments,
    - load and update the training configuration,
    - build the model and datasets,
    - initialize the trainer,
    - launch training.
    """
    # -------------------------------------------------------------------------
    # Runtime setup
    # -------------------------------------------------------------------------
    args = parse_learning_args()
    modality = config.Modality(args.modality)
    logger = utils.setup_logging()
    utils.set_seed(args.seed)

    system_spec = create_system(args.system_name)
    run_stamp = (
        args.stamp_run
        if args.stamp_run is not None
        else paths.create_run_stamp(args.latent_dynamics, args.id, logger)
    )

    # -------------------------------------------------------------------------
    # Paths and configuration
    # -------------------------------------------------------------------------
    run_paths = paths.build_run_paths(
        modality=modality.key,
        system_name=args.system_name,
        run_status="interim",
        stamp_run=run_stamp,
    )
    dataset_paths = paths.build_dataset_paths(args.system_name, str(args.dataset_stamp))

    controller_variant = args.controller_variant
    if controller_variant is None and args.state_in_z:
        controller_variant = "position_in_z"

    run_config = TrainingPipelineConfig.load_default(
        system_name=args.system_name,
        modality=modality,
        controller_variant=controller_variant,
    )
    # Apply user overrides before synchronizing dimensions and derived settings.
    run_config.apply_cli_options(args)
    run_config.sync_shared_params(system_spec)
    run_config.resolve_derived_params()
    save_effective_run_config(
        run_config=run_config,
        run_paths=run_paths,
        args=args,
        run_stamp=run_stamp,
        run_status="interim",
    )

    scenario_level, closed_loop_dt, closed_loop_t_sim = (
        run_config.closed_loop_training.require_scenario_generation_params()
    )
    scenario_gen_config = load_scenario_gen_config(
        system_spec.system_name,
        scenario_level,
        closed_loop_dt,
        closed_loop_t_sim,
    )
    scenario_generator = build_scenario_generator(
        system_spec=system_spec,
        cfg=scenario_gen_config,
        seed=run_config.trainer.seed,
    )

    # -------------------------------------------------------------------------
    # Model initialization
    # -------------------------------------------------------------------------
    model = build_model(modality, run_config.model)
    training_ctx = build_training_context(
        modality=modality,
        run_paths=run_paths,
        model=model,
        trainer_config=run_config.trainer,
    )

    # -------------------------------------------------------------------------
    # Sensor/state-input dataset preparation
    # -------------------------------------------------------------------------
    sensor_dataset_builder = SensorBuilder(
        dataset_paths,
        run_config.data_preparation,
        system_spec.system_dim,
        args.seed,
    )
    processed_states_inputs = sensor_dataset_builder.processed

    scalers = DatasetScalers(
        x=sensor_dataset_builder.x_scaler, u=sensor_dataset_builder.u_scaler
    )
    scalers.save(run_paths.run_dir)

    # -------------------------------------------------------------------------
    # Dataset loader selection
    # -------------------------------------------------------------------------
    if modality is config.Modality.VISION:
        vision_preparation_config = cast(
            VisionPreparationConfig,
            run_config.data_preparation,
        )
        # Prepare the image memmap files before constructing the vision loaders.
        prepare_vision_dataset(
            data_preparation_config=vision_preparation_config,
            dataset_stamp=args.dataset_stamp,
        )

        # Build the visual dataloaders.
        im_dataset_builder = VisionBuilder(
            dataset_paths=dataset_paths,
            config=vision_preparation_config,
            processed_dataset=processed_states_inputs,
            system=system_spec,
            seed=args.seed,
        )
        data_loaders = im_dataset_builder.pipeline()
    else:
        data_loaders = sensor_dataset_builder.data_loaders

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    evaluator = TrainingEvaluator(
        modality=modality,
        system_spec=system_spec,
        run_config=run_config,
        run_paths=run_paths,
        koop_model=model,
        data_loaders=data_loaders,
        scalers=scalers,
        scenario_generator=scenario_generator,
        plant=build_plant(
            system_specs=system_spec,
            dt=run_config.closed_loop_eval.dt,
        ),
    )

    trainer = Trainer(
        modality=modality,
        system_spec=system_spec,
        run_config=run_config,
        run_paths=run_paths,
        koop_model=model,
        training_ctx=training_ctx,
        data_loaders=data_loaders,
        scalers=scalers,
        model_evaluator=evaluator,
    )
    trainer.train_model()


if __name__ == "__main__":
    main()
