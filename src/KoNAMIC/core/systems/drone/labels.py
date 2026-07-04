from __future__ import annotations

from typing import List


def get_u_labels(drone_dim: int) -> List[str]:
    if drone_dim == 1:
        return [r"$F$ [N]"]
    elif drone_dim == 2:
        return [r"$F$ [N]", r"$\tau$ [N$\cdot$m]"]
    elif drone_dim == 3:
        return [r"$F$ [N]", r"$\tau$ [N$\cdot$m]", "", ""]
    raise ValueError(f"Invalid drone dimension {drone_dim}")


def get_x_labels(drone_dim: int, only_positions: bool) -> List[str]:
    if drone_dim == 1:
        if only_positions:
            label = [
                r"$x$ [m]",
            ]
        else:
            label = [
                r"$x$ [m]",
                r"$\dot{x}$ [m/s]",
            ]
    elif drone_dim == 2:
        if only_positions:
            label = [
                r"$y$ [m]",
                r"$z$ [m]",
                r"$\theta$ [°]",
            ]
        else:
            label = [
                r"$y$ [m]",
                r"$z$ [m]",
                r"$\theta$ [°]",
                r"$\dot{y}$ [m/s]",
                r"$\dot{z}$ [m/s]",
                r"$\dot{\theta}$ [°/s]",
            ]
    elif drone_dim == 3:
        if only_positions:
            label = [
                r"$x$ [m]",
                r"$y$ [m]",
                r"$z$ [m]",
                r"$\phi$ [°]",
                r"$\theta$ [°]",
                r"$\psi$ [°]",
            ]
        else:
            label = [
                r"$x$ [m]",
                r"$y$ [m]",
                r"$z$ [m]",
                r"$\phi$ [°]",
                r"$\theta$ [°]",
                r"$\psi$ [°]",
                r"$\dot{x}$ [m/s]",
                r"$\dot{y}$ [m/s]",
                r"$\dot{z}$ [m/s]",
                r"$\dot{\phi}$ [°/s]",
                r"$\dot{\theta}$ [°/s]",
                r"$\dot{\psi}$ [°/s]",
            ]
    else:
        raise ValueError(f"Invalid drone dimension {drone_dim}")
    return label


def get_x_names(drone_dim: int, only_positions: bool) -> list[str]:
    if drone_dim == 1:
        if only_positions:
            names = [
                "x",
            ]
        else:
            names = [
                "x",
                "x_dot",
            ]

    elif drone_dim == 2:
        if only_positions:
            names = [
                "y",
                "z",
                "theta",
            ]
        else:
            names = [
                "y",
                "z",
                "theta",
                "y_dot",
                "z_dot",
                "theta_dot",
            ]

    elif drone_dim == 3:
        if only_positions:
            names = [
                "x",
                "y",
                "z",
                "phi",
                "theta",
                "psi",
            ]
        else:
            names = [
                "x",
                "y",
                "z",
                "phi",
                "theta",
                "psi",
                "x_dot",
                "y_dot",
                "z_dot",
                "phi_dot",
                "theta_dot",
                "psi_dot",
            ]

    else:
        raise ValueError(f"Invalid drone dimension {drone_dim}")

    return names