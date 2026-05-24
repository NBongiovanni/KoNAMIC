import numpy as np

from .config import TrajectoryResult, DataGenerationConfig
from .references import ref6_to_ref12


def simulate_trajectory(
    *,
    cfg: DataGenerationConfig,
    plant,
    controller,
    x0: np.ndarray,
    ref6: np.ndarray,
    time: np.ndarray,
) -> TrajectoryResult:
    n_steps = len(time)

    states = np.zeros((n_steps, 12), dtype=float)
    inputs = np.zeros((n_steps, 4), dtype=float)
    states_ref = ref6.copy()

    states[0] = x0

    ref12 = ref6_to_ref12(ref6)

    controller.reset()
    controller.set_reference(ref12)
    controller.set_initial_conditions(x0)

    x_k = x0.copy()

    for k in range(1, n_steps):
        # Le contrôleur existant gère lui-même son index interne.
        u_k = controller.compute_control({"x_k": x_k})

        x_next = plant.update_state(x_k, u_k)

        states[k] = x_next
        inputs[k] = u_k

        # Récupération des consignes internes phi/theta si disponibles.
        if hasattr(controller, "x_ref_traj") and controller.x_ref_traj is not None:
            kk = min(k - 1, controller.x_ref_traj.shape[0] - 1)
            states_ref[k, 4] = controller.x_ref_traj[kk, 3]
            states_ref[k, 5] = controller.x_ref_traj[kk, 4]

        x_k = x_next

    return TrajectoryResult(
        states=states,
        inputs=inputs,
        states_ref=states_ref,
        time=time,
    )