def get_sys_dimensions(sys_dim: int, task: str = "control") -> tuple[int, int, int]:
    if sys_dim == 1:
        x_dim = 2
        u_dim = 1
        if task == "control":
            x_ref_dim = 1
        else:
            x_ref_dim = x_dim
    elif sys_dim == 2:
        x_dim = 6
        u_dim = 2
        if task == "control":
            x_ref_dim = 2
        else:
            x_ref_dim = x_dim
    elif sys_dim == 3:
        x_dim = 12
        u_dim = 4
        if task == "control":
            x_ref_dim = 3
        else:
            x_ref_dim = x_dim
    else:
        raise ValueError(f"Drone dimension {sys_dim} not supported.")
    return x_dim, u_dim, x_ref_dim


def get_num_views(drone_dim: int) -> int:
    if drone_dim == 2:
        num_views = 1
    elif drone_dim == 3:
        num_views = 2
    else:
        raise ValueError(f"Invalid drone dimension {drone_dim}")
    return num_views
