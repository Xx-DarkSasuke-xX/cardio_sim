from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import linalg


@dataclass(frozen=True)
class ObservabilityReport:
    """
    Summary of observability checks for an LTI system (A, C).

    For n=2, the observability matrix is O = [C; C A].
    More generally, O = [C; C A; ...; C A^{n-1}].
    """
    O: np.ndarray
    rank: int
    det: float | None
    cond: float


def observability_matrix(A: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Build the observability matrix O for (A, C).

    A: (n,n)
    C: (p,n)

    Returns:
        O: (p*n, n)
    """
    A = np.asarray(A, dtype=float)
    C = np.asarray(C, dtype=float)

    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("A must be square.")
    if C.shape[1] != n:
        raise ValueError("C must have shape (p, n).")

    blocks = []
    CAk = C.copy()
    for _ in range(n):
        blocks.append(CAk)
        CAk = CAk @ A

    return np.vstack(blocks)


def observability_checks(A: np.ndarray, C: np.ndarray, tol: float | None = None) -> ObservabilityReport:
    """
    Compute rank, determinant (if square), and condition number of the observability matrix.

    tol:
        Optional tolerance passed to np.linalg.matrix_rank.
    """
    O = observability_matrix(A, C)

    rank = int(np.linalg.matrix_rank(O, tol=tol))
    # det only defined if O is square (happens when p=1)
    det_val = float(np.linalg.det(O)) if O.shape[0] == O.shape[1] else None
    cond_val = float(np.linalg.cond(O))

    return ObservabilityReport(O=O, rank=rank, det=det_val, cond=cond_val)


def observability_gramian_continuous(A: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Continuous-time observability Gramian W_o solves:
        A^T W_o + W_o A + C^T C = 0

    This exists (unique positive semidefinite) if A is Hurwitz (stable).
    """
    A = np.asarray(A, dtype=float)
    C = np.asarray(C, dtype=float)

    Q = C.T @ C
    # Solve A^T W + W A = -Q
    W = linalg.solve_continuous_lyapunov(A.T, -Q)
    return W


def observability_gramian_eigs(A: np.ndarray, C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (W_o, eigvals(W_o)) sorted descending.
    """
    W = observability_gramian_continuous(A, C)
    eigs = np.linalg.eigvalsh(W)  # symmetric eigs
    eigs_sorted = np.sort(eigs)[::-1]
    return W, eigs_sorted


def is_observable(A: np.ndarray, C: np.ndarray, tol: float | None = None) -> bool:
    """
    True iff rank(O) == n.
    """
    O = observability_matrix(A, C)
    n = np.asarray(A).shape[0]
    r = int(np.linalg.matrix_rank(O, tol=tol))
    return r == n
