from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Callable, Optional

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import casadi as ca
import numpy as np

from KoNAMIC.core.control.mpc_solver.mpc_problem import MPCProblem
from KoNAMIC.core.control.config import KmpcControllerConfig


@contextmanager
def _pushd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


class _ReferenceTemplate:
    """
    Template minimal pour capturer une référence d'état de prédiction
    sur l'horizon MPC.

    Convention attendue :
        template["_tvp", k, "state_ref"] = ...
    """

    def __init__(self, N: int, state_dim: int):
        self.N = N
        self.state_dim = state_dim
        self.data = np.zeros((N + 1, state_dim))

    def __setitem__(self, key, value):
        _, k, name = key
        if name != "state_ref":
            return

        arr = np.asarray(value).reshape(-1)
        self.data[k, : arr.shape[0]] = arr

    def __getitem__(self, key):
        _, k, name = key
        if name != "state_ref":
            raise KeyError(name)

        return self.data[k]


@dataclass
class _AcadosInternals:
    ocp: AcadosOcp
    solver: AcadosOcpSolver


class AcadosMPCSolver:
    """
    Solveur MPC basé sur acados.

    Responsabilités :
    - construire un problème OCP acados à partir d'un MPCProblem ;
    - poser la condition initiale à chaque pas MPC ;
    - mettre à jour les références horizonisées ;
    - résoudre le problème MPC ;
    - retourner la première commande optimale ;
    - gérer le reset du solveur entre deux simulations.

    Cette classe ne doit pas :
    - encoder des observations ;
    - faire du scaling ;
    - contenir de logique spécifique Koopman côté contrôleur.
    """

    def __init__(self, controller_dir: Path, controller_config: KmpcControllerConfig):
        self.controller_config = controller_config
        self.solver_options = controller_config.solver_options
        self.control_run_dir = controller_dir

        self.problem: Optional[MPCProblem] = None
        self.reference_provider: Optional[Callable[[float, Any], Any]] = None

        self.sqp_iters: list[int] = []
        self.solver_statuses: list[int] = []
        self._k: int = 0

        self._internals: Optional[_AcadosInternals] = None

        self.state_dim: Optional[int] = None
        self.u_dim: Optional[int] = None
        self.N: Optional[int] = None
        self.dt: Optional[float] = None

        self._u_prev: Optional[np.ndarray] = None

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def build(self, problem: MPCProblem) -> None:
        """
        Construit le solveur acados à partir d'un MPCProblem.
        """
        self.problem = problem
        self.state_dim = problem.state_dim
        self.u_dim = problem.u_dim
        self.N = problem.N
        self.dt = problem.dt
        self.reference_provider = problem.reference_provider

        ocp = AcadosOcp()
        model = AcadosModel()

        x = ca.MX.sym("x", self.state_dim, 1)
        u = ca.MX.sym("u", self.u_dim, 1)

        if problem.f_discrete is None:
            raise ValueError("MPCProblem.f_discrete must be set before building AcadosMPCSolver.")

        x_next = problem.f_discrete(x, u)

        model.x = x
        model.u = u
        model.disc_dyn_expr = x_next
        model.name = "mpc_discrete_model"
        ocp.model = model

        self._setup_solver_options(ocp)
        self._setup_cost(ocp, problem)
        self._setup_terminal_cost(ocp, problem)
        self._setup_input_constraints(ocp, problem)
        self._setup_initial_state_constraint(ocp)

        build_dir = self.control_run_dir
        json_path = str(build_dir / "acados_ocp.json")

        try:
            ocp.code_export_directory = str(build_dir)
        except AttributeError:
            pass

        with _pushd(build_dir):
            solver = AcadosOcpSolver(
                ocp,
                json_file=json_path,
                verbose=False,
            )

        self._internals = _AcadosInternals(ocp=ocp, solver=solver)

        self._initialize_control_guess(solver, problem)

    def set_reference_provider(self, provider: Optional[Callable[[float, Any], Any]]) -> None:
        """
        Enregistre un provider optionnel de référence / TVP sur l'horizon.
        """
        self.reference_provider = provider

    def set_initial_condition(
        self,
        prediction_state_0: np.ndarray,
        u_guess: Optional[np.ndarray] = None,
    ) -> None:
        """
        Définit la condition initiale du problème MPC.
        """
        solver = self._get_solver()
        self._check_is_built()

        x0 = np.asarray(prediction_state_0, dtype=float).reshape(-1)

        if x0.shape != (self.state_dim,):
            raise ValueError(
                f"prediction_state_0 must have shape ({self.state_dim},), got {x0.shape}"
            )

        solver.set(0, "x", x0)

        for i in range(1, self.N + 1):
            solver.set(i, "x", x0)

        if u_guess is not None:
            u_guess = np.asarray(u_guess, dtype=float).reshape(-1)

            if u_guess.shape != (self.u_dim,):
                raise ValueError(
                    f"u_guess must have shape ({self.u_dim},), got {u_guess.shape}"
                )

            for i in range(self.N):
                solver.set(i, "u", u_guess)

    def make_step(self, prediction_state_k: np.ndarray) -> np.ndarray:
        """
        Résout un pas MPC et retourne la première commande optimale.
        """
        solver = self._get_solver()
        self._check_is_built()

        problem = self.problem

        x_k = np.asarray(prediction_state_k, dtype=float).reshape(-1)

        if x_k.shape != (self.state_dim,):
            raise ValueError(
                f"prediction_state_k must have shape ({self.state_dim},), got {x_k.shape}"
            )

        solver.set(0, "x", x_k)

        if self._uses_rate_penalty(problem):
            for i in range(self.N):
                solver.set(i, "p", self._u_prev)

        self._update_reference_if_needed(solver, problem)

        solver.constraints_set(0, "lbx", x_k)
        solver.constraints_set(0, "ubx", x_k)
        solver.set(0, "x", x_k)

        status = solver.solve()
        sqp_iter = solver.get_stats("sqp_iter")
        self.sqp_iters.append(sqp_iter)

        self.solver_statuses.append(int(status))

        u0 = solver.get(0, "u")
        u0 = np.asarray(u0, dtype=float).reshape(-1)

        if self._uses_rate_penalty(problem):
            self._u_prev = u0.copy()

        self._k += 1

        return u0

    def reset(self) -> None:
        """
        Reset du solveur pour une nouvelle simulation indépendante.
        """
        self._k = 0
        self.sqp_iters.clear()
        self.solver_statuses.clear()

        if self._internals is None:
            return

        solver = self._internals.solver
        solver.reset(reset_qp_solver_mem=1)

        if self.problem is not None and self.problem.u_guess is not None:
            u0 = np.asarray(self.problem.u_guess, dtype=float).reshape(-1)

            for i in range(self.N):
                solver.set(i, "u", u0)

            self._u_prev = u0.copy()

        elif self.u_dim is not None:
            self._u_prev = np.zeros(self.u_dim)

    # ---------------------------------------------------------------------
    # Setup helpers
    # ---------------------------------------------------------------------
    def _setup_solver_options(self, ocp: AcadosOcp) -> None:
        ocp.solver_options.N_horizon = self.N
        ocp.solver_options.tf = self.N * self.dt
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"

        nlp_solver_type = self.solver_options.nlp_solver_type
        ocp.solver_options.nlp_solver_type = nlp_solver_type
        ocp.solver_options.print_level = self.solver_options.print_level

        if nlp_solver_type == "SQP_RTI":
            ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
            ocp.solver_options.qp_solver_iter_max = self.solver_options.qp_solver_iter_max
            ocp.solver_options.qp_tol = self.solver_options.qp_tol

        elif nlp_solver_type == "SQP":
            ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
            ocp.solver_options.nlp_solver_max_iter = self.solver_options.nlp_solver_iter_max
            ocp.solver_options.nlp_tol = self.solver_options.nlp_tol
            ocp.solver_options.qp_solver_iter_max = self.solver_options.qp_solver_iter_max
            ocp.solver_options.qp_tol = self.solver_options.qp_tol

        else:
            raise ValueError(f"Unsupported acados nlp_solver_type: {nlp_solver_type}")

    def _setup_cost(self, ocp: AcadosOcp, problem: MPCProblem) -> None:
        if self._uses_rate_penalty(problem):
            self._setup_cost_with_rate_penalty(ocp, problem)
        else:
            self._setup_standard_cost(ocp, problem)

    def _setup_standard_cost(self, ocp: AcadosOcp, problem: MPCProblem) -> None:
        """
        Coût standard :
            ||x - x_ref||²_Q + ||u||²_R
        """
        ny = self.state_dim + self.u_dim

        Vx = np.zeros((ny, self.state_dim))
        Vx[: self.state_dim, : self.state_dim] = np.eye(self.state_dim)

        Vu = np.zeros((ny, self.u_dim))
        Vu[self.state_dim :, : self.u_dim] = np.eye(self.u_dim)

        W = self._blkdiag(problem.Q, problem.R)
        yref = np.zeros(ny)

        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.Vx = Vx
        ocp.cost.Vu = Vu
        ocp.cost.W = W
        ocp.cost.yref = yref

        # Stage 0 explicite pour éviter les mismatch ny_0.
        ocp.cost.cost_type_0 = "LINEAR_LS"
        ocp.cost.Vx_0 = Vx
        ocp.cost.Vu_0 = Vu
        ocp.cost.W_0 = W
        ocp.cost.yref_0 = yref

    def _setup_cost_with_rate_penalty(self, ocp: AcadosOcp, problem: MPCProblem) -> None:
        """
        Coût avec pénalité sur la variation de commande :
            ||x - x_ref||²_Q + ||u||²_R + ||u - u_prev||²_S

        On utilise NONLINEAR_LS pour pouvoir introduire u_prev comme paramètre runtime.
        """
        x = ocp.model.x
        u = ocp.model.u

        u_prev = ca.MX.sym("u_prev", self.u_dim, 1)
        ocp.model.p = u_prev

        if problem.u_guess is not None:
            ocp.parameter_values = np.asarray(problem.u_guess, dtype=float).reshape(-1)
        else:
            ocp.parameter_values = np.zeros(self.u_dim)

        y_expr = ca.vertcat(x, u, u - u_prev)

        ny = self.state_dim + 2 * self.u_dim

        W = np.zeros((ny, ny))
        W[: self.state_dim, : self.state_dim] = problem.Q
        W[
            self.state_dim : self.state_dim + self.u_dim,
            self.state_dim : self.state_dim + self.u_dim,
        ] = problem.R
        W[self.state_dim + self.u_dim :, self.state_dim + self.u_dim :] = problem.S

        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.model.cost_y_expr = y_expr
        ocp.cost.W = W
        ocp.cost.yref = np.zeros(ny)

    def _setup_terminal_cost(self, ocp: AcadosOcp, problem: MPCProblem) -> None:
        ocp.cost.cost_type_e = "LINEAR_LS"
        ocp.cost.Vx_e = np.eye(self.state_dim)
        ocp.cost.W_e = problem.Qf
        ocp.cost.yref_e = np.zeros(self.state_dim)

    def _setup_input_constraints(self, ocp: AcadosOcp, problem: MPCProblem) -> None:
        if not problem.use_input_constraints:
            return

        if problem.u_min is None or problem.u_max is None:
            raise ValueError(
                "MPCProblem.use_input_constraints=True requires u_min and u_max."
            )

        ocp.constraints.lbu = np.asarray(problem.u_min, dtype=float).reshape(-1)
        ocp.constraints.ubu = np.asarray(problem.u_max, dtype=float).reshape(-1)
        ocp.constraints.idxbu = np.arange(self.u_dim, dtype=int)

    def _setup_initial_state_constraint(self, ocp: AcadosOcp) -> None:
        ocp.constraints.idxbx_0 = np.arange(self.state_dim, dtype=int)
        ocp.constraints.lbx_0 = np.zeros(self.state_dim)
        ocp.constraints.ubx_0 = np.zeros(self.state_dim)

    def _initialize_control_guess(
        self,
        solver: AcadosOcpSolver,
        problem: MPCProblem,
    ) -> None:
        if problem.u_guess is not None:
            u0 = np.asarray(problem.u_guess, dtype=float).reshape(-1)

            if u0.shape != (self.u_dim,):
                raise ValueError(
                    f"MPCProblem.u_guess must have shape ({self.u_dim},), got {u0.shape}"
                )

            for i in range(self.N):
                solver.set(i, "u", u0)

            self._u_prev = u0.copy()

        else:
            self._u_prev = np.zeros(self.u_dim)

    # ---------------------------------------------------------------------
    # Runtime helpers
    # ---------------------------------------------------------------------
    def _update_reference_if_needed(
        self,
        solver: AcadosOcpSolver,
        problem: MPCProblem,
    ) -> None:
        if self.reference_provider is None:
            return

        template = _ReferenceTemplate(self.N, self.state_dim)
        t_now = self._k * self.dt

        _ = self.reference_provider(t_now, template)
        x_refs = template.data

        if self._uses_rate_penalty(problem):
            ny = self.state_dim + 2 * self.u_dim
        else:
            ny = self.state_dim + self.u_dim

        for i in range(self.N):
            yref = np.zeros(ny)
            yref[: self.state_dim] = x_refs[i]
            solver.set(i, "yref", yref)

        solver.set(self.N, "yref", x_refs[self.N])

    def _get_solver(self) -> AcadosOcpSolver:
        if self._internals is None:
            raise RuntimeError("AcadosMPCSolver must be built before use.")

        return self._internals.solver

    def _check_is_built(self) -> None:
        if self.problem is None:
            raise RuntimeError("AcadosMPCSolver must be built before use.")

        missing = {
            "state_dim": self.state_dim,
            "u_dim": self.u_dim,
            "N": self.N,
            "dt": self.dt,
        }

        for name, value in missing.items():
            if value is None:
                raise RuntimeError(f"AcadosMPCSolver.{name} is not initialized.")

    @staticmethod
    def _uses_rate_penalty(problem: MPCProblem) -> bool:
        return problem.S is not None and np.any(problem.S > 0)

    @staticmethod
    def _blkdiag(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A = np.asarray(A)
        B = np.asarray(B)

        shape = (A.shape[0] + B.shape[0], A.shape[1] + B.shape[1])
        out = np.zeros(shape)

        out[: A.shape[0], : A.shape[1]] = A
        out[A.shape[0] :, A.shape[1] :] = B

        return out

    @staticmethod
    def print_acados_stats(solver: AcadosOcpSolver) -> None:
        keys = [
            "status",
            "sqp_iter",
            "qp_iter",
            "qp_status",
            "qp_res_max",
            "qp_obj",
            "time_tot",
            "time_lin",
            "time_qp",
            "res_stat",
            "res_eq",
            "res_ineq",
            "res_comp",
        ]

        print("=== ACADOS Solver Statistics ===")
        for key in keys:
            try:
                print(f"{key:>12s}: {solver.get_stats(key)}")
            except Exception:
                pass
        print("================================")