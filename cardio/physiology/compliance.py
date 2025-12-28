from __future__ import annotations

import numpy as np

from cardio.params.dataclasses import ParameterSet
from cardio.physiology.activation import cycle_time, decc_dt, ecc, tvc_tvr


def clv(t: float | np.ndarray, params: ParameterSet) -> float | np.ndarray:
    """
    Time-varying left ventricular compliance C_LV(t).

    From Eq.59:
      C_LV(t) = [ ( (1/Cmin - 1/Cmax) * e_cc(t) ) + (1/Cmax) ]^{-1}

    Parameters
    ----------
    t : float or np.ndarray
        Absolute time [s].
    params : ParameterSet
        Contains Tcc, Cmin, Cmax.

    Returns
    -------
    C : float or np.ndarray
        Ventricular compliance [mL/mmHg].
    """
    Tcc = params.Tcc
    tau = cycle_time(t, Tcc)
    Tvc, Tvr = tvc_tvr(Tcc)

    e = ecc(tau, Tvc, Tvr, Tcc)

    A = (1.0 / params.Cmin) - (1.0 / params.Cmax)
    B = 1.0 / params.Cmax

    C = 1.0 / (A * e + B)

    if np.isscalar(t):
        return float(np.asarray(C).item())
    return np.asarray(C, dtype=float)


def dclv_dt(t: float | np.ndarray, params: ParameterSet) -> float | np.ndarray:
    """
    Time derivative of ventricular compliance dC_LV/dt.

    Using:
      C = 1 / (A*e + B)
      dC/dt = - A * (de/dt) / (A*e + B)^2

    where:
      A = 1/Cmin - 1/Cmax
      B = 1/Cmax

    Parameters
    ----------
    t : float or np.ndarray
        Absolute time [s].
    params : ParameterSet

    Returns
    -------
    dC : float or np.ndarray
        dC_LV/dt [mL/(mmHg*s)].
    """
    Tcc = params.Tcc
    tau = cycle_time(t, Tcc)
    Tvc, Tvr = tvc_tvr(Tcc)

    e = ecc(tau, Tvc, Tvr, Tcc)
    de = decc_dt(tau, Tvc, Tvr, Tcc)

    A = (1.0 / params.Cmin) - (1.0 / params.Cmax)
    B = 1.0 / params.Cmax

    denom = (A * e + B)
    dC = -(A * de) / (denom ** 2)

    if np.isscalar(t):
        return float(np.asarray(dC).item())
    return np.asarray(dC, dtype=float)


def elv(t: float | np.ndarray, params: ParameterSet) -> float | np.ndarray:
    """
    Ventricular elastance E_LV(t) = 1 / C_LV(t).
    """
    C = clv(t, params)
    E = 1.0 / np.asarray(C, dtype=float)
    if np.isscalar(t):
        return float(E.item())
    return E
