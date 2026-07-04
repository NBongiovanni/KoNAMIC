from __future__ import annotations

from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def parse_args_open_loop_simulation() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run open-loop simulation and render rollouts."
    )

    parser.add_argument(
        "--modality",
        type=str,
        choices=["sensor"],
        required=True,
        help="Input modality.",
    )
    parser.add_argument(
        "--system-name",
        type=str,
        choices=["quadrotor_2d", "quadrotor_3d", "cartpole"],
        required=True,
        help="System name.",
    )
    parser.add_argument(
        "--stamp_run",
        "--stamp-run",
        dest="stamp_run",
        type=str,
        required=True,
        help="Training run stamp to evaluate.",
    )
    parser.add_argument(
        "--run-status",
        type=str,
        choices=["final", "interim"],
        required=True,
        help="Training run status.",
    )
    parser.add_argument(
        "--dataset-stamp",
        type=str,
        required=True,
        help="Dataset stamp used to build the evaluation dataloaders.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        required=True,
        help="Checkpoint epoch to evaluate.",
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
        "--only-position",
        dest="only_position",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render only position states. Use --no-only-position to render all states.",
    )

    return parser.parse_args()
