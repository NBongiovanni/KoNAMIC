from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
from tqdm import tqdm

from KoNAMIC.core.simulation import InputsData, TrajectoryData, ClosedLoopTrajectory


class ControlSimulatorBase(ABC):
    def __init__(self, control_params: dict, plant: Any, controller: Any) -> None:
        self.control_params = control_params
        self.plant = plant
        self.controller = controller

        self.dt = float(control_params["dt"])
        self.num_steps_simulation = int(control_params["num_steps_simulation"])

    def _build_observation(
        self,
        state_k: np.ndarray,
        state_km1: np.ndarray,
        state_km2: np.ndarray,
        u_prev: np.ndarray,
    ) -> dict[str, np.ndarray]:
        return {
            "x_k": np.asarray(state_k, dtype=float).copy(),
            "x_km1": np.asarray(state_km1, dtype=float).copy(),
            "x_km2": np.asarray(state_km2, dtype=float).copy(),
            "u_prev": np.asarray(u_prev, dtype=float).copy(),
        }

    def _run_simulation_loop(self, x_init: np.ndarray):
        x_init = np.asarray(x_init, dtype=float).reshape(-1)
        x_dim = x_init.shape[0]

        x_traj = np.zeros((self.num_steps_simulation, x_dim), dtype=float)
        im_traj = None

        state_k = x_init.copy()
        state_km1 = x_init.copy()
        state_km2 = x_init.copy()
        u_k = np.zeros(self.controller.u_dim, dtype=float)

        x_traj[0] = state_k

        for k in tqdm(range(self.num_steps_simulation - 1)):
            observation = self._build_observation(
                state_k=state_k,
                state_km1=state_km1,
                state_km2=state_km2,
                u_prev=u_k,
            )

            u_k = self.controller.compute_control(observation)

            state_kp1 = self._plant_step(state_k, u_k)
            x_traj[k + 1] = state_kp1

            state_km2 = state_km1.copy()
            state_km1 = state_k.copy()
            state_k = state_kp1.copy()

        return x_traj, im_traj

    def _plant_step(self, state_k: np.ndarray, u_k: np.ndarray) -> np.ndarray:
        """
        Adaptateur vers l'API réelle des plantes du projet.

        API supportée en priorité :
        - plant.update_state(x_k, u_k)

        Compatibilité secondaire :
        - plant.step(x_k, u_k)
        - plant.make_step(x_k, u_k)
        - plant.next_step(x_k, u_k)
        - plant.propagate(x_k, u_k)
        - plant.simulate_step(x_k, u_k)
        - plant.dynamics(x_k, u_k)
        - plant(x_k, u_k)
        """
        state_k = np.asarray(state_k, dtype=float).reshape(-1)
        u_k = np.asarray(u_k, dtype=float).reshape(-1)

        candidate_method_names = [
            "update_state",
            "step",
            "make_step",
            "next_step",
            "propagate",
            "simulate_step",
            "dynamics",
        ]

        for method_name in candidate_method_names:
            if hasattr(self.plant, method_name):
                method = getattr(self.plant, method_name)
                x_next = method(state_k, u_k)
                return np.asarray(x_next, dtype=float).reshape(-1)

        if callable(self.plant):
            x_next = self.plant(state_k, u_k)
            return np.asarray(x_next, dtype=float).reshape(-1)

        raise AttributeError(
            "Unsupported plant API. Expected one of these methods: "
            f"{candidate_method_names}, or a callable plant. "
            f"Available public attributes: "
            f"{[name for name in dir(self.plant) if not name.startswith('_')]}"
        )

    def process_output(
        self,
        x_init: np.ndarray,
        x_traj: np.ndarray,
        x_ref_traj: Optional[np.ndarray],
        z_traj: Optional[np.ndarray],
        z_ref_traj: Optional[np.ndarray],
        im_traj: Optional[np.ndarray],
        im_ref_traj: Optional[np.ndarray],
    ) -> ClosedLoopTrajectory:
        """
        Construit un SimResults générique compatible :
        - PID / LQR : x disponible, z/im éventuellement absents
        - Koopman   : x, z, im potentiellement disponibles
        """
        x_init = np.asarray(x_init, dtype=float).reshape(-1)

        time = np.arange(x_traj.shape[0], dtype=float) * self.dt

        x_data = None
        if x_traj is not None:
            x_ref_arr = self._coerce_reference_like_traj(x_ref_traj, x_traj)
            x_error = None if x_ref_arr is None else x_traj - x_ref_arr
            x_data = TrajectoryData(
                traj=x_traj,
                ref_traj=x_ref_arr,
                error=x_error,
            )

        z_data = TrajectoryData(
            traj=self._ensure_2d_or_empty(z_traj),
            ref_traj=self._coerce_reference_like_traj(z_ref_traj, z_traj),
            error=self._compute_error_if_possible(z_traj, z_ref_traj),
        )

        im_data = TrajectoryData(
            traj=im_traj,
            ref_traj=im_ref_traj,
            error=None,
        )

        u_physical = self._extract_u_physical()
        u_scaled = self._extract_u_scaled(u_physical)

        inputs_data = InputsData(
            u_physical=u_physical,
            u_scaled=u_scaled,
        )

        return ClosedLoopTrajectory(
            time=time,
            x_init=x_init,
            x_data=x_data,
            z_data=z_data,
            im_data=im_data,
            inputs_data=inputs_data,
        )

    def _extract_u_physical(self) -> np.ndarray:
        if hasattr(self.controller, "u_physical_traj") and len(self.controller.u_physical_traj) > 0:
            return np.asarray(self.controller.u_physical_traj, dtype=float)

        if hasattr(self.controller, "u_traj") and len(self.controller.u_traj) > 0:
            return np.asarray(self.controller.u_traj, dtype=float)

        return np.zeros((0, self.controller.u_dim), dtype=float)

    def _extract_u_scaled(self, u_physical: np.ndarray) -> np.ndarray:
        if hasattr(self.controller, "u_scaled_traj") and len(self.controller.u_scaled_traj) > 0:
            return np.asarray(self.controller.u_scaled_traj, dtype=float)

        # Pour PID/LQR sans normalisation, on prend la même chose
        return np.asarray(u_physical, dtype=float)

    def _ensure_2d_or_empty(self, arr: Optional[np.ndarray]) -> np.ndarray:
        if arr is None:
            return np.zeros((0, 0), dtype=float)

        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            return arr[:, None]
        return arr

    def _coerce_reference_like_traj(
        self,
        ref: Optional[np.ndarray],
        traj: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if ref is None or traj is None:
            return ref

        ref = np.asarray(ref, dtype=float)
        traj = np.asarray(traj, dtype=float)

        if ref.ndim == 1 and traj.ndim == 2 and ref.shape[0] == traj.shape[1]:
            ref = np.repeat(ref[None, :], traj.shape[0], axis=0)

        if ref.ndim == 2 and traj.ndim == 2 and ref.shape[0] != traj.shape[0]:
            n = min(ref.shape[0], traj.shape[0])
            ref = ref[:n]
            if traj.shape[0] != n:
                return ref

        return ref

    def _compute_error_if_possible(
        self,
        traj: Optional[np.ndarray],
        ref: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if traj is None or ref is None:
            return None

        traj = np.asarray(traj, dtype=float)
        ref = np.asarray(ref, dtype=float)

        if traj.ndim == 1:
            traj = traj[:, None]
        if ref.ndim == 1 and traj.ndim == 2 and ref.shape[0] == traj.shape[1]:
            ref = np.repeat(ref[None, :], traj.shape[0], axis=0)

        if traj.shape != ref.shape:
            n = min(traj.shape[0], ref.shape[0])
            traj = traj[:n]
            ref = ref[:n]

        if traj.shape != ref.shape:
            return None

        return traj - ref

    @abstractmethod
    def run(self, x_init: np.ndarray):
        raise NotImplementedError