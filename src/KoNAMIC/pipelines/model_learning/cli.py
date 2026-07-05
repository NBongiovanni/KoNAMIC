import argparse

from KoNAMIC.config import Modality


def build_learning_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Koopman model")
    p.add_argument("--modality", choices=Modality.values(), required=True)
    p.add_argument("--id", type=str)
    p.add_argument("--stamp-run", type=str)
    p.add_argument("--system-name", type=str)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--latent-dynamics", type=str)
    p.add_argument(
        "--controller-variant",
        choices=["position_in_z", "state_in_z", "full_latent", "structured_latent"],
    )
    p.add_argument("--dataset-stamp", type=str)
    p.add_argument(
        "--geom_losses",
        action=argparse.BooleanOptionalAction,
        help="Enable/disable geometric losses (--no-geom_losses if we want to disable this option)."
    )
    p.add_argument("--state_in_z", action=argparse.BooleanOptionalAction)
    return p


def parse_learning_args() -> argparse.Namespace:
    return build_learning_arg_parser().parse_args()
