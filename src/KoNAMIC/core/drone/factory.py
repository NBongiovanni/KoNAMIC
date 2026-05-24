import numpy as np
from .drone_spec import DroneSpec


def build_drone(drone_dim: int) -> DroneSpec:
    if drone_dim == 2:
        return make_planar_quadrotor()

    if drone_dim == 3:
        return make_3d_quadrotor()

    raise ValueError(f"Drone dimension {drone_dim} is not supported.")

def make_planar_quadrotor() -> DroneSpec:
    return DroneSpec(
        name="planar_quadrotor",
        drone_dim=2,
        mass=0.03,
        gravity=9.81,
        arm_length=0.046,
        inertia=np.array([1e-5, 1e-5, 2e-5]),
        u_min=np.array([0.0, -0.01]),
        u_max=np.array([0.6, 0.01]),
    )


def make_3d_quadrotor() -> DroneSpec:
    return DroneSpec(
        name="quadrotor_3d",
        drone_dim=3,
        mass=0.028,
        gravity=9.81,
        arm_length=0.046,
        inertia=np.array([16.6e-6, 16.6e-6, 29.3e-6]),
        u_min=(-1)*np.ones(4) * 0.6,
        u_max=np.ones(4) * 0.6,
    )