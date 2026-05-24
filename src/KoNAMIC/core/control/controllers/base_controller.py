from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class BaseController(ABC):
    """
    Interface commune à tous les contrôleurs.

    Cette classe ne contient que les éléments réellement transverses :
    - dimensions
    - pas de temps
    - référence
    - conditions initiales
    - saturation des commandes
    - logs génériques
    """

    def __init__(
        self,
        dt: float,
        u_dim: int,
        x_dim: Optional[int] = None,
        u_min: Optional[np.ndarray] = None,
        u_max: Optional[np.ndarray] = None,
        name: Optional[str] = None,
    ) -> None:
        self.dt = float(dt)
        self.u_dim = int(u_dim)
        self.x_dim = None if x_dim is None else int(x_dim)
        self.name = name or self.__class__.__name__

        self.u_min = None if u_min is None else np.asarray(u_min, dtype=float).reshape(-1)
        self.u_max = None if u_max is None else np.asarray(u_max, dtype=float).reshape(-1)

        if self.u_min is not None and self.u_min.shape != (self.u_dim,):
            raise ValueError(f"u_min must have shape ({self.u_dim},), got {self.u_min.shape}")
        if self.u_max is not None and self.u_max.shape != (self.u_dim,):
            raise ValueError(f"u_max must have shape ({self.u_dim},), got {self.u_max.shape}")

        self.reference: Any = None
        self.x_init: Optional[np.ndarray] = None
        self.is_built: bool = False

        self.u_traj: list[np.ndarray] = []
        self.info_traj: list[dict[str, Any]] = []

    def build(self) -> None:
        """
        Hook optionnel pour les contrôleurs qui ont besoin d'une phase de build.
        """
        self.is_built = True

    def reset(self) -> None:
        self.u_traj.clear()
        self.info_traj.clear()
        self.reference = None
        self.x_init = None

    def set_reference(self, reference: Any) -> None:
        self.reference = reference

    def set_initial_conditions(self, x_init: np.ndarray) -> None:
        """
        Méthode appelée par le simulateur avant le début d'un rollout.
        Par défaut, on stocke simplement l'état initial.
        """
        x_init = np.asarray(x_init, dtype=float).reshape(-1)

        if self.x_dim is not None and x_init.shape != (self.x_dim,):
            raise ValueError(f"x_init must have shape ({self.x_dim},), got {x_init.shape}")

        self.x_init = x_init.copy()

    @abstractmethod
    def compute_control(self, *args, **kwargs) -> np.ndarray:
        """
        Calcule la commande courante.
        """
        raise NotImplementedError

    def _clip_control(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float).reshape(-1)
        if u.shape != (self.u_dim,):
            raise ValueError(f"Control must have shape ({self.u_dim},), got {u.shape}")

        if self.u_min is not None:
            u = np.maximum(u, self.u_min)
        if self.u_max is not None:
            u = np.minimum(u, self.u_max)
        return u

    def _store_control(self, u: np.ndarray, info: Optional[dict[str, Any]] = None) -> np.ndarray:
        u = np.asarray(u, dtype=float).reshape(-1)
        self.u_traj.append(u.copy())
        self.info_traj.append(info or {})
        return u