import argparse


def build_dataset_generation_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a dataset from simulated drone trajectories and optional rendered vision data_generation."
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
        type=int,
        help="Drone dimension, e.g. 2 for planar quadrotor.",
    )

    parser.add_argument(
        "--data-config",
        type=str,
        required=True,
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