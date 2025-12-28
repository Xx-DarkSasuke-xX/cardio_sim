from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import signal

from cardio.params.dataclasses import ParameterSet


@dataclass(frozen=True)
class ArterialLTI:
    """
    Linearized arterial Windkessel sub-model in deviation variables.

    States:
        x = [Δp1, ΔQ2]^T
    Input:
        u = ΔQin
    Output:
        y = Δp1

    Matrices:
        xdot = A x + B u
        y    = C x + D u
    """
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray

    # Transfer function coefficients:
    # H(s) = (b1*s + b0) / (s^2 + a1*s + a0)
    a0: float
    a1: float
    b0: float
    b1: float

    # Convenience: scipy systems
    sys_ss: signal.StateSpace
    sys_tf: signal.TransferFunction


def _arterial_resistance(params: ParameterSet) -> float:
    """
    Return the arterial resistance used in the arterial LTI sub-model.

    In our project convention we often set Rcap = Rart -> Rtot = 2*Rart.
    Prefer params.Rtot if present; otherwise fall back to 2*params.Rart.
    """
    rtot = getattr(params, "Rtot", None)
    if rtot is None:
        return 2.0 * float(params.Rart)
    return float(rtot)


def arterial_lti_matrices(params: ParameterSet) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (A,B,C,D) for the linearized arterial sub-model using parameters.

    Uses:
        C_art = params.Cart
        I_art = params.Iart
        R_h   = params.Rtot (preferred) or 2*params.Rart fallback

    Returns:
        A (2x2), B (2x1), C (1x2), D (1x1)
    """
    C_art = float(params.Cart)
    I_art = float(params.Iart)
    R_h = _arterial_resistance(params)

    if C_art <= 0 or I_art <= 0 or R_h <= 0:
        raise ValueError("Arterial LTI requires strictly positive Cart, Iart, and R_h (Rtot).")

    A = np.array(
        [
            [0.0, -1.0 / C_art],
            [1.0 / I_art, -R_h / I_art],
        ],
        dtype=float,
    )
    B = np.array([[1.0 / C_art], [0.0]], dtype=float)
    C = np.array([[1.0, 0.0]], dtype=float)
    D = np.array([[0.0]], dtype=float)
    return A, B, C, D


def arterial_tf_coeffs(params: ParameterSet) -> Tuple[float, float, float, float]:
    """
    Return transfer-function polynomial coefficients (a0,a1,b0,b1) for:

        H(s) = (b1*s + b0) / (s^2 + a1*s + a0)

    with:
        b1 = 1/C
        b0 = R/(C*I)
        a1 = R/I
        a0 = 1/(C*I)

    where R is the arterial resistance R_h (Rtot under our convention).
    """
    C_art = float(params.Cart)
    I_art = float(params.Iart)
    R_h = _arterial_resistance(params)

    if C_art <= 0 or I_art <= 0 or R_h <= 0:
        raise ValueError("TF coefficients require strictly positive Cart, Iart, and R_h (Rtot).")

    b1 = 1.0 / C_art
    b0 = R_h / (C_art * I_art)
    a1 = R_h / I_art
    a0 = 1.0 / (C_art * I_art)
    return a0, a1, b0, b1


def arterial_frequency_params(a0: float, a1: float) -> Tuple[float, float]:
    """
    Return (wn, zeta) from denominator: s^2 + a1*s + a0

        wn   = sqrt(a0)
        zeta = a1 / (2*wn)
    """
    if a0 <= 0:
        raise ValueError("a0 must be > 0 to compute wn.")
    wn = float(np.sqrt(a0))
    zeta = float(a1 / (2.0 * wn))
    return wn, zeta


def arterial_expected_zero(b0: float, b1: float) -> float:
    """
    For H(s) = (b1*s + b0)/(...) -> zero at s0 = -b0/b1.
    """
    if b1 == 0:
        raise ValueError("b1 must be non-zero to compute the zero.")
    return float(-b0 / b1)


def arterial_poles_zeros_from_tf(a0: float, a1: float, b0: float, b1: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute poles and zeros from polynomial coefficients.

    Den: s^2 + a1*s + a0  -> poles are roots
    Num: b1*s + b0        -> single zero (or none if degenerate)
    """
    den = np.array([1.0, float(a1), float(a0)], dtype=float)
    num = np.array([float(b1), float(b0)], dtype=float)

    poles = np.roots(den)

    # For first-order numerator, np.roots gives one root (zero).
    zeros = np.roots(num) if np.any(np.abs(num) > 0) else np.array([], dtype=complex)
    return poles, zeros


def build_arterial_lti(params: ParameterSet) -> ArterialLTI:
    """
    High-level constructor returning an ArterialLTI bundle:
    - A,B,C,D
    - TF coeffs (a0,a1,b0,b1)
    - scipy.signal StateSpace + TransferFunction objects
    """
    A, B, C, D = arterial_lti_matrices(params)
    a0, a1, b0, b1 = arterial_tf_coeffs(params)

    # SciPy transfer function uses descending powers:
    # num = [b1, b0], den = [1, a1, a0]
    sys_tf = signal.TransferFunction([b1, b0], [1.0, a1, a0])
    sys_ss = signal.StateSpace(A, B, C, D)

    return ArterialLTI(
        A=A, B=B, C=C, D=D,
        a0=a0, a1=a1, b0=b0, b1=b1,
        sys_ss=sys_ss, sys_tf=sys_tf,
    )
