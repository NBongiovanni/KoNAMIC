from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from .ref_traj_builder_base import ReferenceTrajBuilderBase


class ReferenceTrajBuilderSensor(ReferenceTrajBuilderBase):
    """
    Builds z_ref directly from sensor/state references.
    """
    def __init__(self, *args, x_scaler: StandardScaler, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_scaler = x_scaler

        if self.koop_model is None:
            raise ValueError("koop_model must not be None for ReferenceTrajBuilderSensor.")

    def build(self):
        if self.verbose:
            print("Generating reference trajectories (sensor)...")
        state_ref_traj = self.build_state_ref()

        # Ici, on suppose que la référence d'état est déjà dans l'espace
        x_ref_traj = np.asarray(state_ref_traj)

        x_scaled = self.x_scaler.transform(x_ref_traj).astype(np.float32)

        device = next(self.koop_model.parameters()).device
        x_t = torch.as_tensor(x_scaled, dtype=torch.float32, device=device)

        with torch.no_grad():
            z_t = self.koop_model.project(x_t)

        z_ref_traj = z_t.detach().cpu().numpy()
        im_ref_traj = None

        return state_ref_traj, im_ref_traj, z_ref_traj[:-1]