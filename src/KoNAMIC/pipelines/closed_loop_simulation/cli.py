import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def parse_args_simulation() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run closed-loop simulations for Koopman or baseline controllers."
    )

    parser.add_argument(
        "--modality",
        type=str,
        choices=["sensor", "vision"],
        help="Input modality.",
    )

    parser.add_argument(
        "--system-name",
        type=str,
        choices=["quadrotor_2d", "quadrotor_3d", "cartpole"],
    )

    parser.add_argument(
        "--scenario-level",
        type=str,
        choices=["smooth", "medium", "aggressive"],
    )

    parser.add_argument("--stamp-run", type=str)
    parser.add_argument(
        "--controller",
        type=str,
        required=True,
        choices=["kmpc", "klqr", "pid", "lqr"],
        help="Controller type.",
    )
    parser.add_argument("--run-status", type=str, choices=["final", "interim"])
    parser.add_argument("--controller-variant", type=str)
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument("--epoch", type=int)
    return parser.parse_args()
