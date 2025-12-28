from __future__ import annotations

import numpy as np

from cardio.params.dataclasses import ParameterSet
from cardio.physiology.compliance import clv, dclv_dt
from cardio.physiology.valves import aortic_flow, mitral_flow


def rhs(t: float, x: np.ndarray, params: ParameterSet) -> np.ndarray:
    """
    Nonlinear systemic circulation model (Eq.56–58).

    State vector (order is fixed across the project):
        x = [pLV, Q2, p1]

    Where:
      - pLV : left ventricular pressure [mmHg]
      - Q2  : peripheral (outflow) arterial flow [mL/s]
      - p1  : aortic/arterial pressure [mmHg]

    Model equations:
      (Eq.56)  C_LV * dpLV/dt = -pLV * dC_LV/dt + P0 - P1
      (Eq.57)  I_art * dQ2/dt = p1 - pRA - R_tot * Q2
      (Eq.58)  C_art * dp1/dt = P1 - Q2

    Valve flows (with smooth Heaviside approximation):
      P0 = (pLA - pLV)/RMV * H(pLA - pLV)
      P1 = (pLV - p1)/RAV * H(pLV - p1)

    Notes:
      - Total resistance uses the project rule: Rtot = 2*Rart (Rcap = Rart).
      - H is smoothed using tanh with slope parameter k_valve.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size != 3:
        raise ValueError("State x must have size 3: [pLV, Q2, p1]")

    pLV, Q2, p1 = x

    # Time-varying ventricular compliance and its derivative
    C_LV = float(clv(t, params))
    dC_LV = float(dclv_dt(t, params))

    # Valve flows (smooth gating)
    k = float(params.k_valve)
    P0 = float(mitral_flow(params.pLA, pLV, params.RMV, k))
    P1 = float(aortic_flow(pLV, p1, params.RAV, k))

    # Eq.56: dpLV/dt
    dpLV_dt = (-pLV * dC_LV + P0 - P1) / C_LV

    # # Eq.57: dQ2/dt
    # dQ2_dt = (p1 - params.pRA - params.Rtot * Q2) / params.Iart
    Rtot = params.Rcap + params.Rart
    dQ2_dt = (p1 - params.pRA - Rtot * Q2) / params.Iart

    # Eq.58: dp1/dt
    dp1_dt = (P1 - Q2) / params.Cart

    return np.array([dpLV_dt, dQ2_dt, dp1_dt], dtype=float)
