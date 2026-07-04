from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np

from KoNAMIC.core.scenarios.scenario import Scenario
from KoNAMIC.core.systems import SystemSpec
from .config import ScenarioGenerationConfig


class ScenarioGenerator(ABC):
    def __init__(
        self,
        system: SystemSpec,
        cfg: ScenarioGenerationConfig,
        seed: int | None = None,
    ):
        self.system = system
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def sample(
        self,
        *,
        time: np.ndarray,
        index: int | None = None,
        num_traj: int | None = None,
    ) -> Scenario:
        raise NotImplementedError

    def generate(self, *, time: np.ndarray, n: int) -> list[Scenario]:
        return [
            self.sample(time=time, index=i, num_traj=n)
            for i in range(n)
        ]

    def iter(self, *, time: np.ndarray, n: int) -> Iterator[Scenario]:
        for i in range(n):
            yield self.sample(time=time, index=i, num_traj=n)

    def _select_profile(
            self,
            *,
            index: int | None = None,
            num_traj: int | None = None,
    ) -> str:
        profiles = self.cfg.scenario.profiles

        if not profiles:
            raise ValueError("No scenario profiles configured.")

        profile_names = list(profiles.keys())
        weights = np.asarray(list(profiles.values()), dtype=float)

        if np.any(weights < 0.0):
            raise ValueError(f"Profile weights must be non-negative, got {profiles}")

        total_weight = float(np.sum(weights))
        if total_weight <= 0.0:
            raise ValueError(
                f"At least one profile weight must be positive, got {profiles}"
            )

        probabilities = weights / total_weight

        if index is None or num_traj is None:
            return str(self.rng.choice(profile_names, p=probabilities))

        if num_traj <= 0:
            raise ValueError(f"num_traj must be positive, got {num_traj}")

        if index < 0 or index >= num_traj:
            raise ValueError(
                f"index must satisfy 0 <= index < num_traj, "
                f"got index={index}, num_traj={num_traj}"
            )

        cumulative = np.cumsum(probabilities)
        relative_position = (index + 0.5) / num_traj

        profile_idx = int(np.searchsorted(cumulative, relative_position, side="right"))
        profile_idx = min(profile_idx, len(profile_names) - 1)

        return str(profile_names[profile_idx])

    @staticmethod
    def _constant_reference(
            *,
            value: np.ndarray,
            time: np.ndarray,
    ) -> np.ndarray:
        value = np.asarray(value, dtype=float).reshape(1, -1)
        return np.repeat(value, repeats=len(time), axis=0)

    @staticmethod
    def _validate_state_reference(
            self,
            *,
            reference: np.ndarray,
            time: np.ndarray,
    ) -> None:
        expected_shape = (len(time), self.system_spec.state_dim)

        if reference.shape != expected_shape:
            raise ValueError(
                f"Invalid reference shape: expected {expected_shape}, "
                f"got {reference.shape}"
            )