import numpy as np
from numpy.linalg import matrix_rank


def ctrb(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Matrice de contrôlabilité C = [B, A B, ..., A^{n-1} B]
    """
    A = np.atleast_2d(np.array(A, dtype=float))
    B = np.atleast_2d(np.array(B, dtype=float))
    n = A.shape[0]
    C = B
    Ak = np.eye(n)
    for _ in range(1, n):
        Ak = Ak @ A
        C = np.concatenate((C, Ak @ B), axis=1)
    return C

def controllability_kalman(A: np.ndarray, B: np.ndarray, tol: float | None = None):
    """
    Test de Kalman : contrôlable ssi rank(C) = n.
    Retourne (controllable: bool, rankC: int, C: ndarray)
    - tol: seuil pour le rang numérique (matrix_rank), None => par défaut NumPy.
    """
    C = ctrb(A, B)
    r = matrix_rank(C, tol=tol)
    n = A.shape[0]
    return (r == n), int(r), C

