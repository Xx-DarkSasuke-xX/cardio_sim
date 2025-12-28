from __future__ import annotations

import numpy as np


def heaviside_smooth(x: float | np.ndarray, k: float) -> float | np.ndarray:
    """
    Smooth approximation of the Heaviside step function using tanh:

        H(x) ≈ (1 + tanh(k x)) / 2

    Parameters
    ----------
    x : float or np.ndarray
        Input value(s).
    k : float
        Slope parameter. Larger k makes the transition sharper.

    Returns
    -------
    H : float or np.ndarray
        Smooth step in [0, 1].
    """
    if k <= 0:
        raise ValueError("k must be > 0")

    x_arr = np.asarray(x, dtype=float)
    H = 0.5 * (1.0 + np.tanh(k * x_arr))

    if np.isscalar(x):
        return float(H.item())
    return H


def mitral_flow(
    pLA: float | np.ndarray,
    pLV: float | np.ndarray,
    RMV: float,
    k: float,
) -> float | np.ndarray:
    """
    Mitral valve inflow P0(t):

        P0 = (pLA - pLV)/RMV * H(pLA - pLV)

    Parameters
    ----------
    pLA, pLV : float or np.ndarray
        Left atrial pressure and left ventricular pressure [mmHg].
    RMV : float
        Mitral valve resistance [mmHg*s/mL].
    k : float
        Smoothing parameter for the Heaviside approximation.

    Returns
    -------
    P0 : float or np.ndarray
        Mitral flow [mL/s].
    """
    if RMV <= 0:
        raise ValueError("RMV must be > 0")

    dp = np.asarray(pLA, dtype=float) - np.asarray(pLV, dtype=float)
    H = heaviside_smooth(dp, k=k)
    P0 = (dp / RMV) * H

    # Preserve scalar if all inputs are scalar
    if np.isscalar(pLA) and np.isscalar(pLV):
        return float(np.asarray(P0).item())
    return np.asarray(P0, dtype=float)


def aortic_flow(
    pLV: float | np.ndarray,
    p1: float | np.ndarray,
    RAV: float,
    k: float,
) -> float | np.ndarray:
    """
    Aortic valve outflow P1(t):

        P1 = (pLV - p1)/RAV * H(pLV - p1)

    Parameters
    ----------
    pLV, p1 : float or np.ndarray
        Left ventricular pressure and aortic pressure [mmHg].
    RAV : float
        Aortic valve resistance [mmHg*s/mL].
    k : float
        Smoothing parameter for the Heaviside approximation.

    Returns
    -------
    P1 : float or np.ndarray
        Aortic flow [mL/s].
    """
    if RAV <= 0:
        raise ValueError("RAV must be > 0")

    dp = np.asarray(pLV, dtype=float) - np.asarray(p1, dtype=float)
    H = heaviside_smooth(dp, k=k)
    P1 = (dp / RAV) * H

    if np.isscalar(pLV) and np.isscalar(p1):
        return float(np.asarray(P1).item())
    return np.asarray(P1, dtype=float)
