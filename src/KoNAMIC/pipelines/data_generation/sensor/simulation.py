import numpy as np

from KoNAMIC.core.drone import DroneSpec
from KoNAMIC.core.plants import Plant
from KoNAMIC.core.control.controllers import BaseController
from .dataset import TrajectoryResult


def simulate_trajectory(
    *,
    drone: DroneSpec,
    plant: Plant,
    controller: BaseController,
    x0: np.ndarray,
    ref_controller: np.ndarray,
    time: np.ndarray,
) -> TrajectoryResult:
    """
    Simule une trajectoire en boucle fermée.

    ref_controller doit déjà être au format attendu par le contrôleur :
        - drone_dim == 2 : shape (T, 6)
        - drone_dim == 3 : shape (T, 12)

    states_ref a la même dimension que states.
    """
    n_steps = len(time)

    x0 = np.asarray(x0, dtype=float).reshape(-1)
    ref_controller = np.asarray(ref_controller, dtype=float)

    if x0.shape != (drone.x_dim,):
        raise ValueError(f"x0 must have shape ({drone.x_dim},), got {x0.shape}")

    if ref_controller.shape != (n_steps, drone.x_dim):
        raise ValueError(
            f"ref_controller must have shape {(n_steps, drone.x_dim)}, "
            f"got {ref_controller.shape}"
        )

    states = np.zeros((n_steps, drone.x_dim), dtype=float)
    inputs = np.zeros((n_steps, drone.u_dim), dtype=float)
    states_ref = ref_controller.copy()

    states[0] = x0

    controller.reset()
    controller.set_reference(ref_controller)
    controller.set_initial_conditions(x0)

    x_k = x0.copy()

    for k in range(1, n_steps):
        u_k = controller.compute_control({"x_k": x_k})
        u_k = np.asarray(u_k, dtype=float).reshape(-1)

        if u_k.shape != (drone.u_dim,):
            raise ValueError(
                f"Controller returned control with shape {u_k.shape}, "
                f"expected ({drone.u_dim},)"
            )

        x_next = plant.update_state(x_k, u_k)
        x_next = np.asarray(x_next, dtype=float).reshape(-1)

        if x_next.shape != (drone.x_dim,):
            raise ValueError(
                f"Plant returned state with shape {x_next.shape}, "
                f"expected ({drone.x_dim},)"
            )

        states[k] = x_next
        inputs[k] = u_k

        if hasattr(controller, "x_ref_traj") and controller.x_ref_traj is not None:
            kk = min(k - 1, controller.x_ref_traj.shape[0] - 1)

            if drone.drone_dim == 2:
                # Convention 2D :
                # state = [y, z, theta, y_dot, z_dot, theta_dot]
                #
                # Le contrôleur planar peut écrire theta_c dans x_ref_traj[:, 2].
                states_ref[k, 2] = controller.x_ref_traj[kk, 2]

            elif drone.drone_dim == 3:
                # Convention 3D :
                # state = [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]
                #
                # Dans votre contrôleur 3D, phi_c et theta_c sont écrits dans
                # x_ref_traj[:, 3] et x_ref_traj[:, 4].
                states_ref[k, 3] = controller.x_ref_traj[kk, 3]
                states_ref[k, 4] = controller.x_ref_traj[kk, 4]

            else:
                raise ValueError(f"Unsupported drone_dim: {drone.drone_dim}")

        x_k = x_next

    return TrajectoryResult(
        states=states,
        inputs=inputs,
        states_ref=states_ref,
        time=time,
    )