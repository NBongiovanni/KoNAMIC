import matplotlib
matplotlib.use("Agg")

from KoNAMIC.core import utils
from KoNAMIC.pipelines.data_preparation import SensorLoader, ImageProcessor, VisionDatasetParams
from KoNAMIC.pipelines.data_generation import ImDatasetGenerator, parse_dataset_generation_args


def main() -> None:
    args = parse_dataset_generation_args()
    project_root = utils.find_project_root()
    vision_dataset_params = VisionDatasetParams.from_yaml(project_root / args.vision_data_config)

    dataset_paths = utils.build_dataset_paths(
        drone_dim=args.drone_dim,
        dataset_stamp=str(args.dataset_stamp),
    )

    vision_dataset_params.image_dataset_dir = dataset_paths.raw_im_dir

    data_loader = SensorLoader(
        dataset_paths,
        args.drone_dim,
        vision_dataset_params.train,
        vision_dataset_params.val_datasets[0],
        vision_dataset_params.val_datasets[1],
        vision_dataset_params.downsample_factor,
    )
    raw_data = data_loader.load_raw_sensor_data()

    splits = [
        ("train", vision_dataset_params.train),
        ("val_1", vision_dataset_params.val_datasets[0]),
        ("val_2", vision_dataset_params.val_datasets[1]),
    ]

    for phase, split_cfg in splits:
        n_traj = raw_data[phase]["x"].shape[0]
        num_steps_pred = split_cfg["num_steps_pred"]
        num_steps_loaded = raw_data[phase]["x"].shape[1]

        im_generator = ImDatasetGenerator(
            vision_dataset_params,
            dataset_paths,
            phase,
            raw_data[phase],
            n_traj,
        )
        im_generator.generate_raw_images()

        im_preprocessor = ImageProcessor(
            vision_dataset_params,
            phase,
            str(args.dataset_stamp),
            num_steps_loaded,
        )
        im_preprocessor.pipeline(
            n_traj,
            vision_dataset_params.resolution,
            num_steps_pred,
        )


if __name__ == '__main__':
    main()