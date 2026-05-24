from __future__ import annotations

from .ref_traj_builder_base import ReferenceTrajBuilderBase


class StateReferenceBuilder(ReferenceTrajBuilderBase):
    """
    Builder générique pour les contrôleurs qui consomment directement
    une trajectoire d'état de référence (LQR, PID, full-state NMPC, etc.).
    """

    def build(self):
        state_ref_traj = self.build_state_ref()

        # Convention alignée avec le cas Koopman :
        # build_state_ref() génère num_steps_simulation + 1 points,
        # alors que le contrôleur utilise en général un horizon de longueur N.
        return state_ref_traj[:-1]