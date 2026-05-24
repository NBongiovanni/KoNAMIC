import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]

def parse_args_comparative_plots() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare several closed-loop simulation results from a YAML preset."
    )
    parser.add_argument(
        "--preset-file",
        type=Path,
        default=PROJECT_ROOT / Path("configs/figures/control.yaml"),
        help="Path to the YAML preset file.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        required=True,
        help="Preset name to load from the YAML file.",
    )
    return parser.parse_args()


def parse_args_simulation() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run closed-loop simulations for Koopman MPC or baseline controllers."
    )

    parser.add_argument(
        "--modality",
        type=str,
        choices=["sensor", "vision"],
        help="Input modality.",
    )
    parser.add_argument(
        "--drone-dim",
        type=int,
        choices=[2, 3],
        help="Drone dimension.",
    )
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument("--case-id", type=int, help="Case ID used for Koopman MPC mode.")
    parser.add_argument(
        "--controller-type",
        type=str,
        choices=["koopman_mpc", "pid", "lqr"],
        help="Controller type.",
    )
    parser.add_argument("--caseid", type=int, help="Case config id.",)
    parser.add_argument("--run-status", type=str)
    return parser.parse_args()
