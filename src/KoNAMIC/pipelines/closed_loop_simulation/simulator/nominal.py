import numpy as np

from .base import ControlSimulatorBase

class NominalControlSimulator(ControlSimulatorBase):
    """
    Simulator in which the plant has the same model as the identified one
    (in the z space).

    Both the controller and the plant use the learned Koopman model dynamics.
    This represents an ideal scenario with perfect model knowledge (no model
    mismatch), useful for evaluating controller performance upper bounds.

    State space: z (Koopman latent sensor)
    """

    def run(self, x_init: np.ndarray):
        # TODO: type hint
        z_0 = self.controller.set_initial_conditions(
            x_init, True
        )

        z_traj, im_traj = self._run_simulation_loop(
            initial_state=z_0,
            update_image_fn=lambda z: self.plant.update_image_from_z_k(np.squeeze(z)),
            nominal_control=True,
        )

        return self.process_output(
            [],
            None, # For nominal control, x trajectories are not available
            self.controller.z_traj,
            self.controller.z_ref_traj,
            im_traj,
            True,
        )