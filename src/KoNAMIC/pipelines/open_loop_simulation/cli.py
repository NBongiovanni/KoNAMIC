from __future__ import annotations

from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def parse_args_comparison():
    parser = argparse.ArgumentParser(
        description="Overlay open-loop simulation results for several stored model configurations."
    )
    parser.add_argument(
        "--preset",
        type=str,
        required=True,
        help="Name of the overlay preset to load from the YAML file.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--preset-file",
        type=Path,
        default=PROJECT_ROOT / Path("configs/pipelines/eval_comparisons/open_loop.yaml"),
        help="Path to the YAML file containing overlay figures.",
    )
    return parser.parse_args()


def parse_args_open_loop_simulation() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run open-loop simulation and render rollouts."
    )

    parser.add_argument(
        "--modality",
        type=str,
        choices=["sensor", "vision"],
        required=True,
        help="Input modality.",
    )
    parser.add_argument(
        "--caseid",
        type=int,
        required=True,
        help="Case identifier.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3,
        help="Random seed.",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="val_2",
        help="Dataset phase.",
    )
    parser.add_argument(
        "--num-steps",
        dest="num_steps",
        type=int,
        required=True,
        help="Number of rollout steps.",
    )
    parser.add_argument(
        "--num-rollouts",
        dest="num_rollouts",
        type=int,
        default=30,
        help="Number of rollouts to render.",
    )
    parser.add_argument(
        "--drone-dim",
        dest="drone_dim",
        type=int,
        help="Drone dimension, only relevant for sensor modality if needed.",
    )
    parser.add_argument(
        "--data_generation-version",
        dest="dataset_version",
        type=int,
        default=None,
        help="Dataset version, only relevant for vision modality.",
    )
    parser.add_argument(
        "--only-position",
        dest="only_position",
        action="store_true",
        help="Render only positions.",
    )
    parser.add_argument(
        "--render-vision",
        dest="render_images",
        action="store_true",
        help="Render image trajectories when supported.",
    )
    parser.add_argument(
        "--snapshots",
        action="store_true",
        help="Enable snapshots rendering.",
    )

    parser.set_defaults(
        only_position=True,
        snapshots=False,
    )

    return parser.parse_args()