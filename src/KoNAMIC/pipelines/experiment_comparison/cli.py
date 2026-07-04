from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def parse_args_comparison(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
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
        default=PROJECT_ROOT / Path("configs/registries/comparisons"),
        help="Path to the comparison preset registry root.",
    )
    return parser.parse_args()
