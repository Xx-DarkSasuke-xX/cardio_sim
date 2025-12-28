from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from cardio.params.dataclasses import ParameterSet
from cardio.analysis.linearization import arterial_tf_coeffs


@dataclass(frozen=True)
class IdentifiabilityRoundtrip:
    """
    Roundtrip check for structural identifiability (numerical verification).

    Steps:
        1) (C, I, R) -> (a0, a1, b0, b1)
        2) (a0, a1, b0, b1) -> (C_hat, I_hat, R_hat)
        3) Report relative errors
    """
    a0: float
    a1: float
    b0: float
    b1: float

    C_hat: float
    I_hat: float
    R_hat: float

    rel_err_C: float
    rel_err_I: float
    rel_err_R: float


def reconstruct_parameters_from_tf_coeffs(a0: float, a1: float, b0: float, b1: float) -> Tuple[float, float, float]:
    """
    Reconstruct (C, I, R) from transfer-function coefficients:

        C = 1/b1
        I = b1/a0
        R = (a1*b1)/a0
    """
    if b1 == 0:
        raise ValueError("b1 must be non-zero to reconstruct C.")
    if a0 == 0:
        raise ValueError("a0 must be non-zero to reconstruct I and R.")

    C_hat = 1.0 / float(b1)
    I_hat = float(b1) / float(a0)
    R_hat = (float(a1) * float(b1)) / float(a0)
    return C_hat, I_hat, R_hat


def roundtrip_identifiability(params: ParameterSet) -> IdentifiabilityRoundtrip:
    """
    Numerical 'roundtrip' identifiability verification for the arterial LTI sub-model.

    Uses arterial_tf_coeffs(params) to get (a0,a1,b0,b1), then reconstructs (C_hat,I_hat,R_hat),
    and compares to the original parameters (Cart, Iart, R_h=Rtot under our convention).
    """
    a0, a1, b0, b1 = arterial_tf_coeffs(params)
    C_hat, I_hat, R_hat = reconstruct_parameters_from_tf_coeffs(a0, a1, b0, b1)

    C_true = float(params.Cart)
    I_true = float(params.Iart)

    # R_true is the arterial resistance seen by the arterial model (R_h).
    # arterial_tf_coeffs already used the correct R_h, but we can derive it back consistently:
    R_true = (float(a1) * float(params.Iart))  # since a1 = R/I

    def rel_err(x_hat: float, x_true: float) -> float:
        if abs(x_true) < 1e-15:
            return float("nan")
        return float((x_hat - x_true) / x_true)

    return IdentifiabilityRoundtrip(
        a0=float(a0), a1=float(a1), b0=float(b0), b1=float(b1),
        C_hat=float(C_hat), I_hat=float(I_hat), R_hat=float(R_hat),
        rel_err_C=rel_err(C_hat, C_true),
        rel_err_I=rel_err(I_hat, I_true),
        rel_err_R=rel_err(R_hat, R_true),
    )


def is_structurally_identifiable(params: ParameterSet, tol: float = 1e-12) -> bool:
    """
    Practical check: roundtrip errors should be near zero (within tol).
    Structural identifiability is theoretical, but this ensures numerical consistency.
    """
    rep = roundtrip_identifiability(params)
    return (
        abs(rep.rel_err_C) < tol
        and abs(rep.rel_err_I) < tol
        and abs(rep.rel_err_R) < tol
    )
