#!/usr/bin/env python
import joblib

import matplotlib
matplotlib.use("Agg")

from KoNAMIC.core import utils
from KoNAMIC.core.drone import DroneSpec
from KoNAMIC.core.models import init_model
from KoNAMIC.pipelines.data_preparation import VisionBuilder, SensorBuilder, prepare_vision_memmap
from KoNAMIC.pipelines.model_learning import Trainer, TrainingConfig, parse_learning_args


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
    logger = utils.setup_logging()
    utils.set_seed(args.seed)

    project_root = utils.find_project_root()
    drone_config = f"configs/components/drones/{args.drone_dim}d_quadrotor.yaml"
    drone = DroneSpec.from_yaml(project_root / drone_config)

    # Create a unique identifier for this training run.
    run_stamp = utils.create_run_stamp(args.dynamics, args.id, logger)

    # -------------------------------------------------------------------------
    # Paths and configuration
    # -------------------------------------------------------------------------
    run_paths = utils.build_run_paths(
        modality=args.modality,
        drone_dim=args.drone_dim,
        run_status="interim",
        stamp_run=run_stamp,
    )
    dataset_paths = utils.build_dataset_paths(args.drone_dim,str(args.dataset_stamp))

    run_config = TrainingConfig.load_base_config(
        modality=args.modality,
        drone_dim=args.drone_dim,
    )


    # Synchronize shared parameters, apply CLI overrides, and register run paths
    # inside the configuration object before saving the resolved configuration.
    run_config.sync_shared_params()
    run_config.apply_cli_options(args)
    params = run_config.to_dict()
    utils.save_yaml(params ,run_paths.run_dir)

    # Retrieve the configuration sub-dictionaries
    model_params = run_config.model_params
    training_params = run_config.training_params
    dataset_params = run_config.dataset_params
    control_params = run_config.control_params

    # -------------------------------------------------------------------------
    # Model initialization
    # -------------------------------------------------------------------------
    model, training_ctx = init_model(
        args.modality,
        run_paths,
        model_params,
        training_params
    )

    # -------------------------------------------------------------------------
    # Sensor/state-input dataset preparation
    # -------------------------------------------------------------------------
    # Sensor data are always processed first. In vision mode, they are also used
    # to provide the state/input information associated with image trajectories.
    state_inputs_dataset_builder = SensorBuilder(
        dataset_paths,
        dataset_params,
        args.drone_dim,
    )
    processed_states_inputs = state_inputs_dataset_builder.processed

    # Save the scalers used to normalize inputs and states. They will be needed
    # later for evaluation, simulation, or control.
    u_scaler = state_inputs_dataset_builder.u_scaler
    joblib.dump(u_scaler, run_paths.run_dir / "u_scaler.pkl")

    x_scaler = state_inputs_dataset_builder.x_scaler
    joblib.dump(x_scaler, run_paths.run_dir / "x_scaler.pkl")

    # -------------------------------------------------------------------------
    # Dataset loader selection
    # -------------------------------------------------------------------------
    if args.modality == "vision":
        # Prepare the image memmap files before constructing the vision loaders.
        # This avoids loading all image trajectories directly into memory.
        prepare_vision_memmap(
            num_traj=dataset_params["train"]["num_traj_loaded"],
            dataset_params=dataset_params,
            dataset_stamp=args.dataset_stamp,
        )

        # Build the visual dataloaders.
        im_dataset_builder = VisionBuilder(
            dataset_paths,
            len(dataset_params["val_datasets"]),
            dataset_params["resolution"],
            dataset_params["dataloader"]["batch_size"],
            processed_states_inputs,
            dataset_params["dataloader"]["num_workers"],
            drone,
            args.drone_dim,
            args.seed,
        )
        dataloader = im_dataset_builder.pipeline()

    else:
        # In sensor mode, training directly uses the dataloaders produced by the
        # sensor dataset builder.
        dataloader = state_inputs_dataset_builder.data_loader

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    trainer = Trainer(
        modality=args.modality,
        run_paths=run_paths,
        model_params=model_params,
        control_params=control_params,
        training_params=training_params,
        drone=drone,
        model=model,
        data_loaders=dataloader,
        prediction_horizon=run_config.prediction_horizon,
        x_scaler=state_inputs_dataset_builder.x_scaler,
        u_scaler=state_inputs_dataset_builder.u_scaler,
        training_ctx=training_ctx,
    )
    trainer.train_model()


if __name__ == "__main__":
    main()