import numpy as np

from KoNAMIC.core.simulation import ClosedLoopTrajectory
from .base import ControlSimulatorBase


class RealControlSimulator(ControlSimulatorBase):
    """
    Simulator using the real physical plant.

    Compatible avec :
    - PID
    - LQR (soon)
    - Koopman MPC (in progress)
    - NMPC with full knowledge (soon)
    """

    def run(self, x_init: np.ndarray) -> ClosedLoopTrajectory:
        init_observation = self._build_observation(
            state_k=x_init,
            state_km1=x_init,
            state_km2=x_init,
            u_prev=np.zeros(self.controller.u_dim, dtype=float),
        )
        self.controller.set_initial_conditions(init_observation)

        x_traj, im_traj = self._run_simulation_loop(x_init)
        x_ref_traj = self.controller.x_ref_traj

        z_traj = getattr(self.controller, "z_traj", None)
        if z_traj is None:
            z_traj = getattr(self.controller, "prediction_state_traj", None)

        z_ref_traj = getattr(self.controller, "z_ref_traj", None)
        if z_ref_traj is None:
            z_ref_traj = getattr(self.controller, "reference_traj", None)
        im_ref_traj = getattr(self.controller, "im_ref_traj", None)

        return self.process_output(
            x_init=x_init,
            x_traj=x_traj,
            x_ref_traj=x_ref_traj,
            z_traj=z_traj,
            z_ref_traj=z_ref_traj,
            im_traj=im_traj,
            im_ref_traj=im_ref_traj,
        )