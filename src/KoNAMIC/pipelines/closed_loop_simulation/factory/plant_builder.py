from KoNAMIC.core.plants import LearnedModel, PlanarQuad, Quad3D


def create_plant(drone, controller_type, ctx):
    dt = float(ctx.control_params["dt"])
    use_nominal = bool(ctx.control_params["use_nominal_plant"])

    if use_nominal:
        if controller_type == "koopman_mpc":
            return LearnedModel(dt, ctx.koop_model)
        raise NotImplementedError

    if drone.drone_dim == 2:
        return PlanarQuad(dt, drone)
    if drone.drone_dim == 3:
        return Quad3D(dt, drone)

    raise ValueError(f"Unknown drone_dim={drone.drone_dim}")