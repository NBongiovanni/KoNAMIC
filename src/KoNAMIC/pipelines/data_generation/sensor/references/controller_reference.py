import numpy as np

from KoNAMIC.core.drone import DroneSpec


def build_controller_reference(ref_user: np.ndarray, drone: DroneSpec) -> np.ndarray:
    ref_user = np.asarray(ref_user, dtype=float)

    if ref_user.ndim != 2:
        raise ValueError(
            f"ref_user must be 2D, got shape {ref_user.shape}"
        )

    n_steps = ref_user.shape[0]
    ref_controller = np.zeros((n_steps, drone.x_dim), dtype=float)

    if drone.drone_dim == 2:
        ref_controller[:, 0] = ref_user[:, 0]
        ref_controller[:, 1] = ref_user[:, 1]
        return ref_controller

    if drone.drone_dim == 3:
        ref_controller[:, 0] = ref_user[:, 0]
        ref_controller[:, 1] = ref_user[:, 1]
        ref_controller[:, 2] = ref_user[:, 2]
        return ref_controller
    raise ValueError(f"Unsupported drone_dim: {drone.drone_dim}")