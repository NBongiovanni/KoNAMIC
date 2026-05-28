from pathlib import Path

from KoNAMIC.core import utils
from KoNAMIC.core.plants import build_quad_plant
from KoNAMIC.core.drone import build_drone

from KoNAMIC.pipelines.data_generation import (
    parse_dataset_generation_args,
    build_controller_factory,
    generate_all_splits,
    SensorGenerationConfig
)


def main():
    args = parse_dataset_generation_args()
    project_root = utils.find_project_root()
    logger = utils.setup_logging()

    yaml_config = utils.load_yaml(project_root / args.data_config)
    config = SensorGenerationConfig.from_dict(yaml_config)
    controller_cfg = utils.load_yaml(project_root / yaml_config["controller"]["config_path"])

    dataset_stamp = Path(utils.make_timestamp(logger))
    dataset_paths = utils.build_dataset_paths(args.drone_dim, str(dataset_stamp))
    drone = build_drone(args.drone_dim)
    plant = build_quad_plant(drone=drone, dt=config.dt)

    controller_factory = build_controller_factory(drone, controller_cfg)

    generate_all_splits(
        cfg=config,
        drone=drone,
        plant=plant,
        controller_factory=controller_factory,
        output_path=dataset_paths.sensor_dir,
        plot_debug=True,
    )


if __name__ == "__main__":
    main()