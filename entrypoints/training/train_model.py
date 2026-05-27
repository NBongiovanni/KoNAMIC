#!/usr/bin/env python
import joblib

import matplotlib
matplotlib.use("Agg")

from KoNAMIC.core import utils
from KoNAMIC.core.drone import build_drone
from KoNAMIC.core.models import init_model
from KoNAMIC.pipelines.data_preparation import VisionBuilder, SensorBuilder
from KoNAMIC.pipelines.model_learning import Trainer, TrainingConfig, parse_learning_args


def main() -> None:
    """
    Training of a Koopman model with a visual dataset.
    """
    args = parse_learning_args()
    logger = utils.setup_logging()
    utils.set_seed(args.seed)
    drone = build_drone(args.drone_dim)
    run_stamp = utils.create_run_stamp(args.dynamics, args.id, logger)

    run_paths = utils.build_run_paths(
        modality=args.modality,
        drone_dim=args.drone_dim,
        run_status="interim",
        stamp_run=run_stamp,
    )
    run_config = TrainingConfig.load_base_config(
        name=args.config, modality=args.modality, drone_dim=args.drone_dim
    )
    dataset_paths = utils.build_dataset_paths(
        drone_dim=args.drone_dim,
        dataset_stamp=str(args.dataset_stamp),
    )
    run_config.sync_shared_params()
    run_config.apply_cli_options(args)
    run_config.define_paths(run_paths)
    params = run_config.to_dict()
    utils.save_yaml(params)

    model_params = run_config.model_params
    training_params = run_config.training_params
    dataset_params = run_config.dataset_params
    control_params = run_config.control_params

    model, training_ctx = init_model(args.modality, model_params, training_params)

    state_inputs_dataset_builder = SensorBuilder(
        dataset_paths,
        dataset_params,
        args.drone_dim
    )
    processed_states_inputs = state_inputs_dataset_builder.processed

    u_scaler = state_inputs_dataset_builder.u_scaler
    joblib.dump(u_scaler, run_paths.run_dir / "u_scaler.pkl")
    x_scaler = state_inputs_dataset_builder.x_scaler
    joblib.dump(x_scaler, run_paths.run_dir / "x_scaler.pkl")

    if args.modality == "vision":
        im_dataset_builder = VisionBuilder(
            dataset_paths,
            len(dataset_params["val_datasets"]),
            dataset_params["resolution"],
            dataset_params["batch_size"],
            processed_states_inputs,
            dataset_params["num_workers"],
            drone,
            args.drone_dim,
            args.seed,
        )
        im_dataset_loader = im_dataset_builder.pipeline()
        dataloader = im_dataset_loader
    else:
        dataloader = state_inputs_dataset_builder.data_loader

    trainer = Trainer(
        modality=args.modality,
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
        run_dir=run_paths.run_dir,
    )
    trainer.train_model()


if __name__ == '__main__':
    main()