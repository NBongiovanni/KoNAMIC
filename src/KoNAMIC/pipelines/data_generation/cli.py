import argparse


def build_sensor_dataset_generation_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a data_generation from simulated drone trajectories "
                    "and optional rendered vision data_generation."
    )

    parser.add_argument(
        "--controller",
        type=str,
        choices=["pid", "lqr"],
        required=True,
    )

    parser.add_argument(
        "--system-name",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    return parser


def build_vision_dataset_generation_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--system-name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dataset-stamp",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--system-dim",
        type=int,
        required=True,
    )

    return parser


def parse_sensor_dataset_generation_args() -> argparse.Namespace:
    return build_sensor_dataset_generation_arg_parser().parse_args()


def parse_vision_dataset_generation_args() -> argparse.Namespace:
    return build_vision_dataset_generation_arg_parser().parse_args()