from KoNAMIC.core.plants import Quad2D, Quad3D


def create_plant(drone, controller_type, ctx):
    dt = float(ctx.control_params["dt"])

    if drone.drone_dim == 2:
        return Quad2D(dt, drone)
    if drone.drone_dim == 3:
        return Quad3D(dt, drone)

    raise ValueError(f"Unknown drone_dim={drone.drone_dim}")