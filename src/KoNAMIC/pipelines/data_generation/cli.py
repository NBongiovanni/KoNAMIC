import argparse


def build_dataset_generation_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a dataset from simulated drone trajectories and optional rendered vision data."
    )

    parser.add_argument(
        "--modality",
        type=str,
        choices=["sensor", "vision"],
        required=True,
        help="Dataset modality to generate.",
    )

    parser.add_argument(
        "--drone-dim",
        dest="drone_dim",
        type=int,
        help="Drone dimension, e.g. 2 for planar quadrotor.",
    )

    parser.add_argument(
        "--dt",
        type=float,
        help="Sampling time used for dataset generation.",
    )

    parser.add_argument(
        "--sensor-data-config",
        dest="sensor_data_config",
        type=str,
        required=True,
        help="Path to the sensor data generation configuration file.",
    )

    parser.add_argument(
        "--vision-data-config",
        dest="vision_data_config",
        type=str,
        required=False,
        help="Path to the vision data generation configuration file.",
    )

    parser.add_argument(
        "--dataset-stamp",
        type=str,
        required=False,
        help="Dataset version name. If omitted, a timestamp may be used.",
    )

    return parser


def parse_dataset_generation_args() -> argparse.Namespace:
    return build_dataset_generation_arg_parser().parse_args()