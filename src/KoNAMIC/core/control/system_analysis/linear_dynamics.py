#!/usr/bin/env python
from __future__ import annotations
from typing import Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.linalg import matrix_rank, eig
from scipy.linalg import solve_discrete_are
import torch

from .linear_algebra_utils import plot_discrete_eigs, save_discrete_eigs

def analyse_linear_dynamics(A: torch.Tensor, B: torch.Tensor) -> None:
    discrete_lti = DiscreteLTI(A.detach().cpu().numpy(), B.detach().cpu().numpy())
    print(f"Is the system controllable? {discrete_lti.is_controllable()}")
    # Q = np.eye(np.shape(A)[0])
    # R = np.eye(2)
    # print(discrete_lti.dlqr(Q, R))

def analyze_AB(A: torch.Tensor, B: torch.Tensor, save_dir: Optional[Path] = None):
    # A: (nz,nz), B: (nz,nu)
    nz, nu = B.shape
    device = A.device
    N=20

    # Eigenvalues of A
    eigA = torch.linalg.eigvals(A).cpu()
    abs_eigA = eigA.abs()
    rho = abs_eigA.max().item()

    # Column norms of B
    col_norms = torch.linalg.norm(B, dim=0).cpu()

    # Singular values of B
    sB = torch.linalg.svdvals(B).cpu()
    condB = (sB.max() / (sB.min() + 1e-12)).item()

    # Controllability Gramian over horizon N
    W = torch.zeros((nz, nz), device=device, dtype=A.dtype)
    Ak = torch.eye(nz, device=device, dtype=A.dtype)
    BBt = B @ B.T
    for _ in range(N):
        W = W + Ak @ BBt @ Ak.T
        Ak = A @ Ak

    eigW = torch.linalg.eigvalsh(W).cpu()
    # Avoid negative numerical noise
    eigW = torch.clamp(eigW, min=0.0)
    condW = (eigW.max() / (eigW.min() + 1e-12)).item()

    out = {
        "rho(A)": rho,
        "|eig(A)| min/median/max": (abs_eigA.min().item(), abs_eigA.median().item(), abs_eigA.max().item()),
        "B column norms": col_norms.tolist(),
        "sv(B) min/median/max": (sB.min().item(), sB.median().item(), sB.max().item()),
        "cond(B)": condB,
        "eig(W_N) min/median/max": (eigW.min().item(), eigW.median().item(), eigW.max().item()),
        "cond(W_N)": condW,
    }
    print(out)
    analyse_linear_dynamics(A, B)

    save_dir.mkdir(parents=True, exist_ok=True)

    plot_discrete_eigs(
        A.detach().cpu().numpy(),
        title="Eigenvalues – bilinear Koopman model",
        savepath=str(save_dir / "spectrum_bilinear.pdf"),
    )
    save_discrete_eigs(A.detach().cpu().numpy(), save_dir / "A_eigs.npz")
    return out


@dataclass
class DiscreteLTI:
    """Système linéaire discret : x_{k+1} = Ad x_k + Bd u_k + cd
    - Ad: (n,n)
    - Bd: (n,m)
    - cd: (n,) optionnel (terme affine, par ex. gravité non compensée)
    """
    Ad: np.ndarray
    Bd: np.ndarray
    cd: Optional[np.ndarray] = None

    def __post_init__(self):
        self.Ad = np.atleast_2d(np.asarray(self.Ad, dtype=float))
        self.Bd = np.atleast_2d(np.asarray(self.Bd, dtype=float))
        n, n2 = self.Ad.shape
        nB, m = self.Bd.shape
        if n != n2 or nB != n:
            raise ValueError("Dimensions incompatibles: Ad doit être (n,n) et Bd (n,m).")
        if self.cd is None:
            self.cd = np.zeros(n)
        else:
            self.cd = np.asarray(self.cd, dtype=float).reshape(n)
        self.n, self.m = n, m

    # ---------- Contrôlabilité ----------
    def controllability_matrix(self, horizon: Optional[int] = None) -> np.ndarray:
        """Matrice de contrôlabilité C = [B, AB, A^2B, ..., A^{n-1}B]
        - horizon: nb de blocs; par défaut n (Kalman).
        """
        h = self.n if horizon is None else int(horizon)
        blocks = [self.Bd]
        Ak = np.eye(self.n)
        for _ in range(1, h):
            Ak = Ak @ self.Ad
            blocks.append(Ak @ self.Bd)
        return np.concatenate(blocks, axis=1)

    def controllability_rank(self, tol: float = 1e-10, horizon: Optional[int] = None) -> int:
        C = self.controllability_matrix(horizon=horizon)
        return matrix_rank(C, tol=tol)

    def is_controllable(self, tol: float = 1e-10) -> bool:
        return self.controllability_rank(tol=tol, horizon=self.n) == self.n

    # ---------- DLQR / bouclage ----------
    def dlqr(self, Q: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """DLQR (DARE): minimise sum x_k^T Q x_k + u_k^T R u_k
        Retourne (K, P, poles) avec u_k = -K x_k si vous régulez autour de l'origine.
        Pour réguler autour d'un (x*,u*): u = u* - K (x - x*).
        """
        Q = np.atleast_2d(np.asarray(Q, dtype=float))
        R = np.atleast_2d(np.asarray(R, dtype=float))
        if Q.shape != (self.n, self.n):
            raise ValueError(f"Q doit être ({self.n},{self.n}).")
        if R.shape != (self.m, self.m):
            raise ValueError(f"R doit être ({self.m},{self.m}).")

        P = solve_discrete_are(self.Ad, self.Bd, Q, R)
        # Formule DLQR “standard”
        S = R + self.Bd.T @ P @ self.Bd
        K = np.linalg.solve(S, self.Bd.T @ P @ self.Ad)
        poles = eig(self.Ad - self.Bd @ K)[0]
        return K, P, poles

    def closed_loop_poles(self, K: np.ndarray) -> np.ndarray:
        """Valeurs propres de (Ad - Bd K)."""
        K = np.atleast_2d(np.asarray(K, dtype=float))
        if K.shape != (self.m, self.n):
            raise ValueError(f"K doit être ({self.m},{self.n}).")
        return eig(self.Ad - self.Bd @ K)[0]