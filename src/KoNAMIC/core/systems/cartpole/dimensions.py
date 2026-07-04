def get_cartpole_dimensions(task: str = "control") -> tuple[int, int, int]:
    """
    Return state, control and reference dimensions for a standard cart-pole.

    State convention used here:
        x = [p, theta, p_dot, theta_dot]

    where p is the cart position and theta is the pole angle.
    This positions-first convention is consistent with DroneSpec.split_state().
    """
    x_dim = 4
    u_dim = 1

    if task == "control":
        x_ref_dim = 2  # [p_ref, theta_ref]
    elif task == "open_loop":
        x_ref_dim = x_dim
    else:
        raise ValueError(f"Unsupported task {task!r}. Expected 'control' or 'open_loop'.")

    return x_dim, u_dim, x_ref_dim


def get_num_views() -> int:
    return 1
