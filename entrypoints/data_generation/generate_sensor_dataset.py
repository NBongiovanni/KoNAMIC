from pathlib import Path

from KoNAMIC.core import utils
from KoNAMIC.core.plants import build_quad_plant
from KoNAMIC.core.drone import DroneSpec

from KoNAMIC.pipelines.data_generation import (
    parse_dataset_generation_args,
    build_controller_factory,
    generate_all_dataset_splits,
    SensorGenerationConfig,
)


def main():
    # ------------------------------------------------------------------
    # Parse command-line arguments and initialize project utilities.
    # ------------------------------------------------------------------
    args = parse_dataset_generation_args()
    project_root = utils.find_project_root()
    logger = utils.setup_logging()

    # ------------------------------------------------------------------
    # Load the sensor data generation configuration.
    # ------------------------------------------------------------------
    data_gen_config_path = (
        f"configs/pipelines/data_generation/"
        f"{args.modality}_{args.drone_dim}d.yaml"
    )
    yaml_config = utils.load_yaml(project_root / data_gen_config_path)
    data_gen_config = SensorGenerationConfig.from_dict(yaml_config)

    # ------------------------------------------------------------------
    # Load the controller configuration used during trajectory generation.
    # ------------------------------------------------------------------
    control_config_path = (
        f"configs/components/controllers/pid/"
        f"{args.modality}_{args.drone_dim}d_base.yaml"
    )
    controller_cfg = utils.load_yaml(project_root / control_config_path)

    # ------------------------------------------------------------------
    # Create a new dataset output directory.
    # ------------------------------------------------------------------
    dataset_stamp = Path(utils.make_timestamp(logger))
    dataset_paths = utils.build_dataset_paths(args.drone_dim, str(dataset_stamp))

    # ------------------------------------------------------------------
    # Build the drone specification and the corresponding physical plant.
    # ------------------------------------------------------------------
    drone_config = f"configs/components/drones/{args.drone_dim}d_quadrotor.yaml"
    drone = DroneSpec.from_yaml(project_root / drone_config)

    plant = build_quad_plant(drone=drone, dt=data_gen_config.dt)
    # ------------------------------------------------------------------
    # Build a controller factory.
    # ------------------------------------------------------------------
    controller_factory = build_controller_factory(drone, controller_cfg)

    # ------------------------------------------------------------------
    # Generate all dataset splits.
    # ------------------------------------------------------------------
    generate_all_dataset_splits(
        cfg=data_gen_config,
        drone=drone,
        plant=plant,
        controller_factory=controller_factory,
        output_path=dataset_paths.sensor_dir,
        plot_debug=True,
    )


if __name__ == "__main__":
    main()