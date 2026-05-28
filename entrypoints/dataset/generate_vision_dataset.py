import matplotlib
matplotlib.use("Agg")

from KoNAMIC.core import utils
from KoNAMIC.pipelines.data_preparation import SensorLoader, SensorLoadSpec
from KoNAMIC.pipelines.data_generation import (
    VisionDatasetRenderer,
    parse_dataset_generation_args,
    VisionGenerationConfig,
)


def main() -> None:
    args = parse_dataset_generation_args()
    project_root = utils.find_project_root()
    config = VisionGenerationConfig.from_yaml(project_root / args.data_config)
    dataset_paths = utils.build_dataset_paths(
        args.drone_dim,
        args.dataset_stamp,
    )

    split_specs = {
        "train": SensorLoadSpec(num_steps_loaded=None),
        "val_1": SensorLoadSpec(num_steps_loaded=None),
        "val_2": SensorLoadSpec(num_steps_loaded=None),
    }

    data_loader = SensorLoader(
        dataset_paths=dataset_paths,
        drone_dim=args.drone_dim,
        split_specs=split_specs,
        downsample_factor=config.render_stride,
    )

    raw_data = data_loader.load_raw_sensor_data()

    for phase, phase_data in raw_data.items():
        im_generator = VisionDatasetRenderer(
            params=config,
            dataset_path=dataset_paths,
            phase=phase,
            raw_data=phase_data,
        )
        im_generator.generate_raw_images()


if __name__ == "__main__":
    main()