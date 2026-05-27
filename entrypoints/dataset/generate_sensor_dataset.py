from pathlib import Path

from KoNAMIC.core import utils
from KoNAMIC.core.plants import build_quad_plant
from KoNAMIC.core.drone import build_drone

from KoNAMIC.pipelines.data_generation import (
    parse_dataset_generation_args,
    build_controller_factory,
    generate_all_splits,
    SensorDatasetParams
)

def load_generation_config(sensor_data_config: str, project_root: Path) -> tuple[SensorDatasetParams, dict]:
    yaml_path = project_root / sensor_data_config
    yaml_cfg = utils.load_yaml(yaml_path)

    cfg = SensorDatasetParams(**yaml_cfg)

    ctrl_path = project_root / yaml_cfg["config_controller"]
    ctrl_cfg = utils.load_yaml(ctrl_path)
    return cfg, ctrl_cfg


def make_output_dir(
    *,
    project_root: Path,
    logger,
    modality: str,
    drone_dim: int,
) -> Path:
    stamp = Path(utils.make_timestamp(logger))
    output_dir = Path("datasets") / f"{drone_dim}d" / stamp / modality
    return project_root / utils.make_unique_dir(output_dir, False)


def main():
    args = parse_dataset_generation_args()

    project_root = utils.find_project_root()
    logger = utils.setup_logging()

    cfg, ctrl_cfg = load_generation_config(args.sensor_data_config, project_root)

    drone = build_drone(args.drone_dim)
    plant = build_quad_plant(drone=drone, dt=cfg.dt)

    controller_factory = build_controller_factory(
        drone=drone,
        cfg=cfg,
        ctrl_cfg=ctrl_cfg,
    )

    output_path = make_output_dir(
        project_root=project_root,
        logger=logger,
        modality=args.modality,
        drone_dim=drone.drone_dim,
    )

    generate_all_splits(
        cfg=cfg,
        drone=drone,
        plant=plant,
        controller_factory=controller_factory,
        output_path=output_path,
        plot_debug=True,
    )


if __name__ == "__main__":
    main()