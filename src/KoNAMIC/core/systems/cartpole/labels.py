from __future__ import annotations

from typing import List


def get_u_labels() -> List[str]:
    return [r"$F$ [N]"]


def get_x_labels(only_positions: bool = False) -> List[str]:
    if only_positions:
        return [
            r"$p$ [m]",
            r"$\theta$ [°]",
        ]

    return [
        r"$p$ [m]",
        r"$\theta$ [°]",
        r"$\dot{p}$ [m/s]",
        r"$\dot{\theta}$ [°/s]",
    ]


def get_x_names(only_positions: bool = False) -> List[str]:
    if only_positions:
        return [
            "p",
            "theta",
        ]

    return [
        "p",
        "theta",
        "p_dot",
        "theta_dot",
    ]
