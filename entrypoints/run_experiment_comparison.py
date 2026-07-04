#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from KoNAMIC import paths, utils
from KoNAMIC.pipelines.experiment_comparison.modes import (
    run_closed_loop_comparison,
    run_open_loop_comparison,
)


matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})


def main() -> None:
    args = _parse_args()
    logger = utils.setup_logging()

    if args.eval_type == "open_loop":
        run_open_loop_comparison(args, logger)
    elif args.eval_type == "closed_loop":
        run_closed_loop_comparison(args, logger)
    else:
        raise ValueError(f"Unsupported eval_type: {args.eval_type!r}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare open-loop or closed-loop results from a YAML preset."
    )
    parser.add_argument(
        "--eval-type",
        choices=("open_loop", "closed_loop"),
        required=True,
        help="Comparison mode to run.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        required=True,
        help="Preset name to load from the comparison registry.",
    )
    parser.add_argument(
        "--system_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--modality",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--preset-file",
        type=Path,
        default=paths.find_project_root() / Path("configs/registries/comparisons"),
        help="Path to the comparison preset registry root.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
