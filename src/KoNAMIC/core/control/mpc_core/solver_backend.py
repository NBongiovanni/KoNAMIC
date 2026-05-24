from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional, Any

import numpy as np

from KoNAMIC.core.control.mpc_core.mpc_problem import MPCProblem


class SolverBackend(ABC):
    """
    Interface minimale d'un backend de solveur MPC.

    Responsabilités :
    - construire le solveur à partir d'un MPCProblem
    - recevoir l'état initial du problème à chaque pas
    - résoudre un pas MPC
    - gérer le reset du solveur
    - éventuellement consommer une référence horizonisée via provider

    Cette classe NE doit PAS :
    - encoder des observations,
    - faire du scaling,
    - contenir de logique spécifique Koopman côté contrôleur.
    """

    def __init__(self) -> None:
        self.problem: Optional[MPCProblem] = None
        self.reference_provider: Optional[Callable[[float, Any], Any]] = None
        self.sqp_iters: list[int] = []
        self._k: int = 0

    @abstractmethod
    def build(self, problem: MPCProblem) -> None:
        """
        Construit les structures internes du solveur à partir du problème MPC.
        """
        raise NotImplementedError

    @abstractmethod
    def set_initial_condition(
        self,
        prediction_state_0: np.ndarray,
        u_guess: Optional[np.ndarray] = None,
    ) -> None:
        """
        Définit la condition initiale du problème (état de prédiction courant).
        """
        raise NotImplementedError

    def set_reference_provider(self, provider: Optional[Callable[[float, Any], Any]]) -> None:
        """
        Enregistre un provider optionnel de référence / TVP sur l'horizon.
        """
        self.reference_provider = provider

    @abstractmethod
    def make_step(self, prediction_state_k: np.ndarray) -> np.ndarray:
        """
        Résout un pas MPC à partir de l'état de prédiction courant
        et retourne la commande optimale au premier pas.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """
        Remet le backend dans un état propre pour une nouvelle simulation.
        """
        raise NotImplementedError