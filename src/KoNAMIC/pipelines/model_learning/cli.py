import argparse


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train/Resume Koopman model (sensors)")
    p.add_argument("--modality", type=str)
    p.add_argument("--id", type=str)
    p.add_argument("--config", type=str)
    p.add_argument("--drone_dim", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dynamics", type=str)
    p.add_argument(
        "--geom_losses",
        action=argparse.BooleanOptionalAction,
        help="Enable/disable geometric losses (--no-geom_losses if we want to disable this option)."
    )
    p.add_argument(
        "--state_in_z",
        action=argparse.BooleanOptionalAction,
    )
    return p
