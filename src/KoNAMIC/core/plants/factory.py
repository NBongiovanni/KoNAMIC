from KoNAMIC.core.systems import DroneSpec, CartPoleSpec
from .quadrotors.quad_2d import Quad2D
from .quadrotors.quad_3d import Quad3D
from .cartpole import CartPole


def build_plant(system_specs, dt: float):
    if isinstance(system_specs, DroneSpec):
        if system_specs.system_dim == 2:
            return Quad2D(dt=dt, system=system_specs)
        if system_specs.system_dim == 3:
            return Quad3D(dt=dt, system=system_specs)
        raise ValueError(f"Unsupported quadrotor sys_dim={system_specs.system_dim}")

    if isinstance(system_specs, CartPoleSpec):
        return CartPole(dt=dt, system=system_specs)

    raise TypeError(
        f"Unsupported system specification type: {type(system_specs).__name__}"
    )
