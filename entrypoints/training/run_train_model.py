#!/usr/bin/env python
from KoNAMIC.core import utils
from KoNAMIC.core.drone import build_drone
from KoNAMIC.core.models import init_koop_model
from KoNAMIC.pipelines.model_learning import (
    Trainer, generate_run_paths, TrainingConfig, build_arg_parser
)
from KoNAMIC.pipelines.data_pipeline import StateInputsDatasetBuilder


def main() -> None:
    args = build_arg_parser().parse_args()
    logger = utils.setup_logging()
    utils.set_seed(args.seed)

    drone = build_drone(args.drone_dim)
    paths = generate_run_paths(
        args.modality,
        args.drone_dim,
        args.dynamics,
        args.id,
        logger
    )
    run_config = TrainingConfig.load_config(
        name=args.config, modality=args.modality, drone_dim=args.drone_dim
    )
    run_config.sync_shared_params()
    run_config.apply_cli_options(args)
    run_config.define_paths(paths)
    params = run_config.to_dict()
    utils.save_config_yaml(params)

    model_params = run_config.model_params
    training_params = run_config.training_params
    dataset_params = run_config.dataset_params
    control_params = run_config.control_params

    model, training_ctx = init_koop_model(
        args.modality, model_params, training_params
    )
    state_dataset_builder = StateInputsDatasetBuilder(dataset_params, args.drone_dim)
    dataloader = state_dataset_builder.data_loader
    state_dataset_builder.save_scalers(paths.run_dir)

    trainer = Trainer(
        modality=args.modality,
        model_params=model_params,
        control_params=control_params,
        training_params=training_params,
        drone=drone,
        model=model,
        data_loaders=dataloader,
        prediction_horizon=run_config.prediction_horizon,
        x_scaler=state_dataset_builder.x_scaler,
        u_scaler=state_dataset_builder.u_scaler,
        training_ctx=training_ctx,
        run_dir=paths.run_dir,
    )
    trainer.train_model()


if __name__ == '__main__':
    main()