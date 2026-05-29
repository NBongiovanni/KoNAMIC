import matplotlib

# Use a non-interactive backend because this script is intended to run
# as a batch dataset generation entry point, possibly on a remote machine.
matplotlib.use("Agg")

from KoNAMIC.core import utils
from KoNAMIC.pipelines.data_preparation import SensorLoader, SensorLoadSpec
from KoNAMIC.pipelines.data_generation import (
    VisionDatasetRenderer,
    parse_dataset_generation_args,
    VisionGenerationConfig,
)


def main() -> None:
    # ------------------------------------------------------------------
    # Parse command-line arguments and locate the project root.
    # ------------------------------------------------------------------
    args = parse_dataset_generation_args()
    project_root = utils.find_project_root()

    # ------------------------------------------------------------------
    # Load the vision generation configuration.
    #
    # This configuration defines how sensor trajectories are converted
    # into rendered image sequences, for example the image resolution and
    # the rendering stride.
    # ------------------------------------------------------------------
    config = VisionGenerationConfig.from_yaml(project_root / args.data_config)

    # ------------------------------------------------------------------
    # Reconstruct the dataset paths from the requested dataset stamp.
    #
    # Unlike sensor generation, vision generation usually starts from an
    # existing sensor dataset. The dataset stamp therefore identifies the
    # already-generated trajectories to render.
    # ------------------------------------------------------------------
    dataset_paths = utils.build_dataset_paths(
        args.drone_dim,
        args.dataset_stamp,
    )

    # ------------------------------------------------------------------
    # Define how many time steps to load for each split.
    #
    # num_steps_loaded=None means that the complete raw sensor trajectory
    # is loaded for each split before rendering the corresponding images.
    # ------------------------------------------------------------------
    split_specs = {
        "train": SensorLoadSpec(num_steps_loaded=None),
        "val_1": SensorLoadSpec(num_steps_loaded=None),
        "val_2": SensorLoadSpec(num_steps_loaded=None),
    }

    # ------------------------------------------------------------------
    # Load raw sensor trajectories.
    #
    # The downsample factor is tied to the render stride so that the image
    # dataset can be rendered at a lower temporal frequency if desired.
    # ------------------------------------------------------------------
    data_loader = SensorLoader(
        dataset_paths=dataset_paths,
        drone_dim=args.drone_dim,
        split_specs=split_specs,
        downsample_factor=config.render_stride,
    )

    raw_data = data_loader.load_raw_sensor_data()

    # ------------------------------------------------------------------
    # Render raw images for each dataset split.
    #
    # Each split is processed independently so that train/validation image
    # folders remain aligned with the corresponding sensor trajectories.
    # ------------------------------------------------------------------
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