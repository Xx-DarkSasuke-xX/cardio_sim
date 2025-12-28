from __future__ import annotations

import numpy as np


def cycle_time(t: float | np.ndarray, Tcc: float) -> float | np.ndarray:
    """
    Map absolute time t to cycle time tau in [0, Tcc).

    Parameters
    ----------
    t : float or np.ndarray
        Absolute time [s].
    Tcc : float
        Cardiac cycle duration [s].

    Returns
    -------
    tau : float or np.ndarray
        Time within the cycle, tau = t mod Tcc.
    """
    if Tcc <= 0:
        raise ValueError("Tcc must be > 0")
    return np.mod(t, Tcc)


def tvc_tvr(Tcc: float, Tcc_ref: float = 6.0 / 7.0) -> tuple[float, float]:
    """
    Compute contraction and relaxation durations (Tvc, Tvr) for the activation function.

    From the report (Eq.63–64):
      Tvc = 0.3 * (Tcc / Tcc_ref)
      Tvr = 0.15 * (Tcc / Tcc_ref)

    Parameters
    ----------
    Tcc : float
        Cardiac cycle duration [s].
    Tcc_ref : float
        Reference cycle duration [s] (default 6/7 s ~ 70 bpm).

    Returns
    -------
    (Tvc, Tvr) : tuple[float, float]
        Ventricular contraction and relaxation times [s].
    """
    if Tcc <= 0:
        raise ValueError("Tcc must be > 0")
    if Tcc_ref <= 0:
        raise ValueError("Tcc_ref must be > 0")

    scale = Tcc / Tcc_ref
    Tvc = 0.3 * scale
    Tvr = 0.15 * scale
    return float(Tvc), float(Tvr)


def ecc(tau: float | np.ndarray, Tvc: float, Tvr: float, Tcc: float) -> float | np.ndarray:
    """
    Normalized activation function e_cc(tau) over one cardiac cycle.

    Piecewise definition (Eq.60–62):
      - Contraction: 0 <= tau <= Tvc
          e = 0.5 * (1 - cos(pi * tau / Tvc))
      - Relaxation: Tvc <= tau <= Tvc + Tvr
          e = 0.5 * (1 + cos(pi * (tau - Tvc) / Tvr))
      - Rest:       Tvc + Tvr <= tau <= Tcc
          e = 0

    Parameters
    ----------
    tau : float or np.ndarray
        Time within cycle [s].
    Tvc, Tvr, Tcc : float
        Contraction time, relaxation time, and cycle duration [s].

    Returns
    -------
    e : float or np.ndarray
        Activation in [0, 1].
    """
    if Tvc <= 0 or Tvr <= 0 or Tcc <= 0:
        raise ValueError("Tvc, Tvr, and Tcc must be > 0")

    tau_arr = np.asarray(tau, dtype=float)
    e = np.zeros_like(tau_arr)

    # contraction
    m1 = (tau_arr >= 0.0) & (tau_arr <= Tvc)
    if np.any(m1):
        e[m1] = 0.5 * (1.0 - np.cos(np.pi * tau_arr[m1] / Tvc))

    # relaxation
    m2 = (tau_arr > Tvc) & (tau_arr <= Tvc + Tvr)
    if np.any(m2):
        e[m2] = 0.5 * (1.0 + np.cos(np.pi * (tau_arr[m2] - Tvc) / Tvr))

    # rest: already zero
    if np.isscalar(tau):
        return float(e.item())
    return e


def decc_dt(tau: float | np.ndarray, Tvc: float, Tvr: float, Tcc: float) -> float | np.ndarray:
    """
    Time derivative of the activation function: d/dt e_cc(tau).

    From Eq.60–62:
      - Contraction: de/dt = (pi/(2*Tvc)) * sin(pi * tau / Tvc)
      - Relaxation:  de/dt = -(pi/(2*Tvr)) * sin(pi * (tau - Tvc) / Tvr)
      - Rest:        de/dt = 0

    Parameters
    ----------
    tau : float or np.ndarray
        Time within cycle [s].
    Tvc, Tvr, Tcc : float
        Contraction time, relaxation time, and cycle duration [s].

    Returns
    -------
    de_dt : float or np.ndarray
        Derivative of activation.
    """
    if Tvc <= 0 or Tvr <= 0 or Tcc <= 0:
        raise ValueError("Tvc, Tvr, and Tcc must be > 0")

    tau_arr = np.asarray(tau, dtype=float)
    de = np.zeros_like(tau_arr)

    # contraction
    m1 = (tau_arr >= 0.0) & (tau_arr <= Tvc)
    if np.any(m1):
        de[m1] = (np.pi / (2.0 * Tvc)) * np.sin(np.pi * tau_arr[m1] / Tvc)

    # relaxation
    m2 = (tau_arr > Tvc) & (tau_arr <= Tvc + Tvr)
    if np.any(m2):
        de[m2] = -(np.pi / (2.0 * Tvr)) * np.sin(np.pi * (tau_arr[m2] - Tvc) / Tvr)

    # rest: already zero
    if np.isscalar(tau):
        return float(de.item())
    return de
