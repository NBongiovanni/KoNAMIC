from __future__ import annotations
import contextlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional
import numpy as np

from KoNAMIC.core.control.controllers.base_controller import BaseController
from KoNAMIC.core.control.mpc_solver.acados_mpc_solver import AcadosMPCSolver
from KoNAMIC.core.control.mpc_solver.mpc_problem import MPCProblem


class MPCControllerBase(BaseController, ABC):
    """
    Base générique pour les contrôleurs MPC/NMPC.

    Cette classe gère :
    - build du problème
    - interface solver/backend
    - horizon
    - coûts / contraintes
    - référence horizonisée
    - condition initiale
    - appel solveur
    - post-traitement de la commande

    Les sous-classes définissent :
    - la nature de l'état de prédiction (x, z, ...)
    - la dynamique utilisée dans le problème
    - le mapping observation -> état de prédiction
    """

    def __init__(
        self,
        dt: float,
        x_dim: Optional[int],
        u_dim: int,
        prediction_dim: int,
        horizon: int,
        solver: AcadosMPCSolver,
        control_runs_dir: Optional[str] = None,
        u_min: Optional[np.ndarray] = None,
        u_max: Optional[np.ndarray] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            dt=dt,
            x_dim=x_dim,
            u_dim=u_dim,
            u_min=u_min,
            u_max=u_max,
            name=name,
        )

        self.prediction_dim = int(prediction_dim)
        self.horizon = int(horizon)
        self.backend = solver
        self.control_runs_dir = None if control_runs_dir is None else Path(control_runs_dir)

        self.problem: Optional[MPCProblem] = None
        self.prediction_state_traj: list[np.ndarray] = []
        self.reference_traj: Optional[np.ndarray] = None

    def reset(self) -> None:
        super().reset()
        # TODO: vérifier que je n'ai pas besoin de remettre à None le self.mpc_problem
        self.prediction_state_traj.clear()
        self.reference_traj = None
        self.backend.reset()

    def set_reference(self, reference: np.ndarray) -> None:
        """
        Référence dans l'espace utilisé par le MPC.
        Typiquement :
        - z_ref pour KoopmanMPCController
        - x_ref pour FullStateNMPCController
        """
        ref = np.asarray(reference, dtype=float)
        if ref.ndim == 1:
            ref = ref[None, :]
        if ref.shape[1] != self.prediction_dim:
            raise ValueError(
                f"Reference must have shape (T, {self.prediction_dim}), got {ref.shape}"
            )

        self.reference_traj = ref
        super().set_reference(ref)

    def build(self) -> None:
        Q, Qf, R = self._build_cost_matrices()
        f_discrete = self._build_prediction_dynamics()
        u_guess = self._get_initial_control_guess()

        self.problem = MPCProblem(
            dt=self.dt,
            N=self.horizon,
            state_dim=self.prediction_dim,
            u_dim=self.u_dim,
            Q=Q,
            Qf=Qf,
            R=R,
            S=self._build_input_rate_matrix(),
            use_input_constraints=self._use_input_constraints(),
            u_min=self.u_min,
            u_max=self.u_max,
            f_discrete=f_discrete,
            reference_provider=self._make_reference_provider(),
            u_guess=u_guess,
        )
        self.backend.build(self.problem)
        self.is_built = True

    def set_initial_conditions(self, observation: Any) -> None:
        if self.problem is None:
            raise RuntimeError("Call build() before set_initial_conditions().")

        s0 = self._observation_to_prediction_state(observation)
        self.prediction_state_traj.append(s0.copy())
        self.backend.set_initial_condition(s0, self.problem.u_guess)

    def compute_control(self, observation: Any) -> np.ndarray:
        if self.problem is None:
            raise RuntimeError("Call build() before compute_control().")

        s_k = self._observation_to_prediction_state(observation)
        self.prediction_state_traj.append(s_k.copy())

        if self.control_runs_dir is not None:
            self.control_runs_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = self.control_runs_dir / "solver_stdout.txt"
            with open(stdout_path, "a") as f, contextlib.redirect_stdout(f):
                u_raw = self.backend.make_step(s_k)
        else:
            u_raw = self.backend.make_step(s_k)

        u = self._postprocess_control(u_raw)
        return self._store_control(u, info={"prediction_state": s_k})

    def _make_reference_provider(self):
        def provider(t_now, template):
            if self.reference_traj is None:
                raise RuntimeError("Call set_reference() before running the controller.")

            k0 = int(float(t_now) / self.dt)

            for k in range(self.horizon + 1):
                idx = min(k0 + k, self.reference_traj.shape[0] - 1)
                template["_tvp", k, "state_ref"] = self.reference_traj[idx]

            return template

        return provider

    def _postprocess_control(self, u_raw: np.ndarray) -> np.ndarray:
        u = np.asarray(u_raw, dtype=float).reshape(-1)
        return self._clip_control(u)

    def _build_input_rate_matrix(self) -> np.ndarray:
        return np.zeros((self.u_dim, self.u_dim))

    def _use_input_constraints(self) -> bool:
        return self.u_min is not None or self.u_max is not None

    def _get_initial_control_guess(self) -> np.ndarray:
        return np.zeros(self.u_dim, dtype=float)

    @abstractmethod
    def _build_cost_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def _build_prediction_dynamics(self):
        raise NotImplementedError

    @abstractmethod
    def _observation_to_prediction_state(self, observation: Any) -> np.ndarray:
        raise NotImplementedError