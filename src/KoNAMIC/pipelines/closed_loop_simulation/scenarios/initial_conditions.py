import numpy as np


def create_state_init_conditions(x_dim: int, control_params: dict) -> np.ndarray:
    x_init = np.zeros(x_dim, dtype=float)
    for j in range(x_dim):
        if control_params["x_init"][j]["rand"]:
            init_min = control_params["x_init"][j]["min"]
            init_max = control_params["x_init"][j]["max"]
            x_init[j] = np.random.uniform(init_min, init_max)
        else:
            x_init[j] = control_params["x_init"][j]["value"]
    print(f"x_init: {x_init}")
    return x_init