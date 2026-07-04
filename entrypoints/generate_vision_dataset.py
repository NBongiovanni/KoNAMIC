import matplotlib
matplotlib.use("Agg")

from KoNAMIC import paths
from KoNAMIC.pipelines.data_preparation import SensorLoader, SensorLoadSpec
from KoNAMIC.pipelines.data_generation import (
    VisionDatasetRenderer,
    parse_vision_dataset_generation_args,
    SensorGenerationConfig,
    VisionGenerationConfig,
)


def main() -> None:
    args = parse_vision_dataset_generation_args()
    project_root = paths.find_project_root()
    run_config = VisionGenerationConfig.from_yaml(
        project_root
        / "configs"
        / "pipelines"
        / "data_generation"
        / f"{args.system_name}"
        / "vision.yaml"
    )

    dataset_paths = paths.build_dataset_paths(
        args.system_name,
        args.dataset_stamp,
    )

    sensor_config_path = dataset_paths.sensor_generation_config
    if not sensor_config_path.exists():
        raise FileNotFoundError(
            f"Missing sensor generation config: {sensor_config_path}. "
            "Regenerate the sensor dataset with the current data-generation pipeline."
        )

    sensor_config = SensorGenerationConfig.from_yaml(sensor_config_path)
    if sensor_config.system_name != args.system_name:
        raise ValueError(
            f"Sensor dataset system_name={sensor_config.system_name!r} does not match "
            f"requested system_name={args.system_name!r}."
        )
    if sensor_config.modality != "sensor":
        raise ValueError(
            f"Expected a sensor generation config, got modality={sensor_config.modality!r}."
        )

    split_specs = {
        "train": SensorLoadSpec(num_steps_loaded=None),
        "val_1": SensorLoadSpec(num_steps_loaded=None),
        "val_2": SensorLoadSpec(num_steps_loaded=None),
    }

    # ------------------------------------------------------------------
    # Load raw sensor trajectories.
    # ------------------------------------------------------------------
    data_loader = SensorLoader(
        dataset_paths=dataset_paths,
        system_dim=args.system_dim,
        split_specs=split_specs,
        downsample_factor=run_config.render_stride,
    )

    raw_data = data_loader.load_raw_sensor_data()

    # ------------------------------------------------------------------
    # Render raw images for each dataset split.
    # ------------------------------------------------------------------
    for phase, phase_data in raw_data.items():
        im_generator = VisionDatasetRenderer(
            config=run_config,
            dataset_path=dataset_paths,
            phase=phase,
            raw_data=phase_data,
        )
        im_generator.generate_raw_images()


if __name__ == "__main__":
    main()