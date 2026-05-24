from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from .linear_algebra_utils import _add_matrix_to_linear_basis, _qr_rank, _as2d, mat_power

# ============================================================
# Convenience API
# ============================================================

def analyse_bilinear_dynamics(
        A: np.ndarray,
        Bs: List[np.ndarray],
        horizon: int = 4,
        max_switches: int = 3,
        max_power: Optional[int] = None,
        trials: int = 5,
        tol: float = 1e-9,
        rng_seed: int = 0,
        use='word'
) -> Dict[str, Any]:
    """
    In this code, all the dynamics is discrete-time.
    Helper that mirrors a simple analyse_* API.
    use: 'word' (default, recommended) or 'commutator' (heuristic).
    """
    sys = DiscreteBilinear(A, Bs)
    if use == 'word':
        res = sys.is_controllable_word(
            horizon=horizon,
            max_switches=max_switches,
            max_power=max_power,
            trials=trials,
            tol=tol,
            rng_seed=rng_seed
        )
    elif use == 'commutator':
        res = sys.is_controllable_commutator(
            max_depth=(max_power or 3),
            trials=trials,
            tol=tol,
            rng_seed=rng_seed
        )
    else:
        raise ValueError("use must be 'word' or 'commutator'")
    print("Controllability analysis of the bilinear system:")
    print(f"[Bilinear-Discrete/{use}] n={sys.n}, m={sys.m} | controllable? {res['is_controllable']} | "
          f"ranks={res['rank_per_trial']} | cfg={res.get('config',{})}")
    print("")
    return res


@dataclass
class DiscreteBilinear:
    """
    Discrete-time bilinear system (without Du):
        x_{k+1} = A x_k + sum_i u_i[k] * B_i x_k
                = (A + sum_i u_i[k] B_i) x_k

    - A: (n,n)
    - Bs: list of (n,n)
    """
    A: np.ndarray
    Bs: List[np.ndarray]

    def __post_init__(self):
        self.A = _as2d(self.A)
        if not isinstance(self.Bs, (list, tuple)) or len(self.Bs) == 0:
            raise ValueError("Bs must be a non-empty list/tuple of (n,n) matrices.")
        self.Bs = [ _as2d(B) for B in self.Bs ]
        n = self.A.shape[0]
        for i, B in enumerate(self.Bs):
            if B.shape != (n, n):
                raise ValueError(f"B[{i}] has shape {B.shape}, expected {(n,n)}.")
        self.n = n
        self.m = len(self.Bs)

    # --------------------------------------------------------
    # Word-based first-order reachability matrices
    # --------------------------------------------------------
    def word_basis(
            self,
            horizon: int = 4,
            max_switches: int = 3,
            max_power: Optional[int] = None,
            tol: float = 1e-12
    ) -> Dict[str, Any]:
        """
        Build a linear basis of matrices of the form:
            A^{t0} B_{i1} A^{t1} B_{i2} ... A^{tp}
        with total length <= horizon, number of Bs <= max_switches,
        and each exponent t_j in [0 .. max_power].
        These matrices correspond to first-order variations wrt inputs
        (evaluated near zero input) that characterize local reachability.

        Returns dict with:
          - 'matrices': list[np.ndarray] (basis)
          - 'basis_size': int
          - 'config': dict of parameters
        """
        if horizon < 0:
            raise ValueError("horizon must be >= 0")
        if max_switches < 0:
            raise ValueError("max_switches must be >= 0")
        if max_power is None:
            max_power = horizon  # safe default

        basis_cols: List[np.ndarray] = []
        matrices: List[np.ndarray] = []

        def try_add(M: np.ndarray) -> bool:
            added = _add_matrix_to_linear_basis(M, basis_cols, tol)
            if added:
                matrices.append(M)
            return added

        # Precompute A^k
        A_powers = [mat_power(self.A, k) for k in range(max_power + 1)]

        # Generate words by DFS with pruning based on length and switches
        # We avoid explicit word tuples; we build matrices directly.

        def extend_words(
                current_M: np.ndarray,
                steps_used: int,
                switches_used: int
        ):
            """
            Try appending either:
              - a terminal block A^t (no extra B),
              - or a block A^t then B_i and continue.
            'steps_used' counts total length; each B adds 1.
            """
            # Terminal A^t (no more B)
            for t in range(0, max_power + 1):
                total_len = steps_used + t
                if total_len > horizon:
                    break
                M_term = current_M @ A_powers[t]
                try_add(M_term)
            if switches_used >= max_switches:
                return

            # Add A^t then one B_i and continue
            for t in range(0, max_power + 1):
                len_after_A = steps_used + t
                if len_after_A > horizon:
                    break
                Ablk = A_powers[t]
                for Bi in self.Bs:
                    total_len = len_after_A + 1  # count the B
                    if total_len > horizon:
                        continue
                    M_next = current_M @ Ablk @ Bi
                    extend_words(M_next, total_len, switches_used + 1)

        # Start from identity (empty word)
        extend_words(np.eye(self.n), steps_used=0, switches_used=0)

        return {
            "matrices": matrices,
            "basis_size": len(basis_cols),
            "config": {
                "horizon": horizon,
                "max_switches": max_switches,
                "max_power": max_power,
            },
        }

    def reachable_rank_at(
            self,
            x0: np.ndarray,
            horizon: int = 4,
            max_switches: int = 3,
            max_power: Optional[int] = None,
            tol: float = 1e-9
    ) -> int:
        """
        Rank of span{ M @ x0 | M in word_basis(...) }.
        """
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        if x0.shape[0] != self.n:
            raise ValueError(f"x0 must have shape ({self.n},). Got {x0.shape}.")
        W = self.word_basis(
            horizon=horizon,
            max_switches=max_switches,
            max_power=max_power,
            tol=max(tol, 1e-12)
        )
        mats = W["matrices"]
        if not mats:
            return 0
        Mx = np.column_stack([M @ x0 for M in mats])
        return _qr_rank(Mx, tol)

    def is_controllable_word(
            self,
            horizon: int = 4,
            max_switches: int = 3,
            max_power: Optional[int] = None,
            trials: int = 5,
            tol: float = 1e-9,
            rng_seed: Optional[int] = 0,
            require_all_trials: bool = False
    ) -> Dict[str, Any]:
        """
        Numerical first-order reachability test using the word basis.
        If require_all_trials=False, returns True if any random x0 achieves rank n.
        """
        rng = np.random.default_rng(rng_seed)
        ranks = []
        W = self.word_basis(
            horizon=horizon,
            max_switches=max_switches,
            max_power=max_power,
            tol=max(tol, 1e-12)
        )
        mats = W["matrices"]
        last_vectors = None
        for _ in range(trials):
            x0 = rng.standard_normal(self.n)
            x0 /= (np.linalg.norm(x0) + 1e-15)
            Mx = np.column_stack([M @ x0 for M in mats]) if mats else np.zeros((self.n, 0))
            rank = _qr_rank(Mx, tol)
            ranks.append(rank)
            last_vectors = [M @ x0 for M in mats]
        ok = (min(ranks) == self.n) if require_all_trials else (max(ranks) == self.n)
        return {
            "is_controllable": bool(ok),
            "rank_per_trial": ranks,
            "n": self.n,
            "num_fields": len(mats),
            "basis_size_mats": W["basis_size"],
            "vectors_last_trial": last_vectors,
            "method": "word",
            "config": W["config"],
            "notes": (
                "Discrete bilinear controllability (first-order, word-based). "
                "Increase horizon/max_switches if needed; complexity grows combinatorially."
            ),
        }

    # --------------------------------------------------------
    # (Optional) Commutator surrogate for discrete-time
    # --------------------------------------------------------
    def commutator_basis(self, max_depth: int = 3, tol: float = 1e-12) -> Dict[str, Any]:
        """
        Heuristic surrogate using commutators [A,·] and [·,·], similar to continuous-time LARC.
        """
        def lie_bracket(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
            return X @ Y - Y @ X

        basis_cols: List[np.ndarray] = []
        matrices: List[np.ndarray] = []

        def try_add(M: np.ndarray) -> bool:
            added = _add_matrix_to_linear_basis(M, basis_cols, tol=tol)
            if added:
                matrices.append(M)
            return added

        for Bi in self.Bs:
            try_add(Bi)

        current = matrices.copy()
        depth_reached = 0
        for depth in range(1, max_depth + 1):
            new_set: List[np.ndarray] = []
            for X in current:
                cand = lie_bracket(self.A, X)
                if try_add(cand):
                    new_set.append(cand)
            mats_snapshot = matrices.copy()
            for X in current:
                for Y in mats_snapshot:
                    cand = lie_bracket(X, Y)
                    if try_add(cand):
                        new_set.append(cand)
            if not new_set:
                depth_reached = depth
                break
            depth_reached = depth
            current = new_set

        return {
            "matrices": matrices,
            "basis_size": len(basis_cols),
            "depth_reached": depth_reached,
        }

    def is_controllable_commutator(
            self,
            max_depth: int = 3,
            trials: int = 5,
            tol: float = 1e-9,
            rng_seed: Optional[int] = 0,
            require_all_trials: bool = False
    ) -> Dict[str, Any]:
        """
        Heuristic test mirroring the continuous-time LARC for discrete-time.
        """
        rng = np.random.default_rng(rng_seed)
        ranks = []
        C = self.commutator_basis(max_depth=max_depth, tol=max(tol, 1e-12))
        mats = C["matrices"]
        last_vectors = None
        for _ in range(trials):
            x0 = rng.standard_normal(self.n)
            x0 /= (np.linalg.norm(x0) + 1e-15)
            Mx = np.column_stack([M @ x0 for M in mats]) if mats else np.zeros((self.n, 0))
            rank = _qr_rank(Mx, tol)
            ranks.append(rank)
            last_vectors = [M @ x0 for M in mats]
        ok = (min(ranks) == self.n) if require_all_trials else (max(ranks) == self.n)
        return {
            "is_controllable": bool(ok),
            "rank_per_trial": ranks,
            "n": self.n,
            "num_fields": len(mats),
            "basis_size_mats": C["basis_size"],
            "vectors_last_trial": last_vectors,
            "method": "commutator",
            "config": {"max_depth": max_depth},
            "notes": (
                "Heuristic commutator-based surrogate; for discrete time prefer 'word' test. "
                "Increase max_depth if needed."
            ),
        }

# ============================================================
# __main__ quick test
# ============================================================

if __name__ == "__main__":
    # Example (2D)
    A = np.array([[0., 1.], [0., 0.]])
    B1 = np.array([[0., 0.], [1., 0.]])  # choose any Bs you want

    print("Word-based test")
    res_w = analyse_bilinear_dynamics(
        A,
        [B1],
        horizon=4,
        max_switches=3,
        trials=5,
        rng_seed=42,
        use='word'
    )
    print(res_w)

    print("\nCommutator surrogate")
    res_c = analyse_bilinear_dynamics(A, [B1], trials=5, rng_seed=42, use='commutator')
    print(res_c)