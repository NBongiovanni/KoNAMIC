from pathlib import Path

from KoNAMIC.core import utils
from KoNAMIC.core.plants import build_quad_plant
from KoNAMIC.core.drone import DroneSpec

from KoNAMIC.pipelines.data_generation import (
    parse_dataset_generation_args,
    build_controller_factory,
    generate_all_splits,
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
    #
    # This file defines the simulation parameters used to generate
    # state/input trajectories, such as dt, number of trajectories,
    # trajectory types, and split-specific settings.
    # ------------------------------------------------------------------
    data_gen_config_path = (
        f"configs/pipelines/data_generation/"
        f"{args.modality}_{args.drone_dim}d.yaml"
    )
    yaml_config = utils.load_yaml(project_root / data_gen_config_path)
    data_gen_config = SensorGenerationConfig.from_dict(yaml_config)

    # ------------------------------------------------------------------
    # Load the controller configuration used during trajectory generation.
    #
    # For sensor dataset generation, trajectories are generated in closed
    # loop using a PID controller.
    # ------------------------------------------------------------------
    control_config_path = (
        f"configs/components/controllers/pid/"
        f"{args.modality}_{args.drone_dim}d_base.yaml"
    )
    controller_cfg = utils.load_yaml(project_root / control_config_path)

    # ------------------------------------------------------------------
    # Create a new dataset output directory.
    #
    # The timestamp is used as the dataset identifier and determines where
    # the generated sensor trajectories will be saved.
    # ------------------------------------------------------------------
    dataset_stamp = Path(utils.make_timestamp(logger))
    dataset_paths = utils.build_dataset_paths(args.drone_dim, str(dataset_stamp))

    # ------------------------------------------------------------------
    # Build the drone specification and the corresponding physical plant.
    #
    # DroneSpec stores the physical parameters, dimensions, labels, and
    # state/control conventions. The plant uses this specification to
    # simulate the quadrotor dynamics.
    # ------------------------------------------------------------------
    drone_config = f"configs/components/drones/{args.drone_dim}d_quadrotor.yaml"
    drone = DroneSpec.from_yaml(project_root / drone_config)

    plant = build_quad_plant(
        drone=drone,
        dt=data_gen_config.dt,
    )

    # ------------------------------------------------------------------
    # Build a controller factory.
    #
    # A factory is used so that each generated trajectory can instantiate
    # a fresh controller when needed.
    # ------------------------------------------------------------------
    controller_factory = build_controller_factory(drone, controller_cfg)

    # ------------------------------------------------------------------
    # Generate all dataset splits.
    #
    # This typically creates train/validation splits and saves the
    # generated state, input, reference, and time arrays.
    # ------------------------------------------------------------------
    generate_all_splits(
        cfg=data_gen_config,
        drone=drone,
        plant=plant,
        controller_factory=controller_factory,
        output_path=dataset_paths.sensor_dir,
        plot_debug=True,
    )


if __name__ == "__main__":
    main()