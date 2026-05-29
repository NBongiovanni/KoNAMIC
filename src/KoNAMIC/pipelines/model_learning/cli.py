import argparse


def build_learning_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Koopman model")
    p.add_argument("--modality", type=str)
    p.add_argument("--id", type=str)
    p.add_argument("--drone_dim", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dynamics", type=str)
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
