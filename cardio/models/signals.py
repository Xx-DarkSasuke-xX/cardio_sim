from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from cardio.params.dataclasses import ParameterSet
from cardio.physiology.compliance import clv, dclv_dt, elv
from cardio.physiology.valves import aortic_flow, mitral_flow


def reconstruct_signals(
    t: np.ndarray,
    x: np.ndarray,
    params: ParameterSet,
) -> Dict[str, np.ndarray]:
    """
    Reconstruct derived signals from simulation states.

    Inputs
    ------
    t : (N,) array
        Time vector [s].
    x : (N,3) array
        State trajectory with columns [pLV, Q2, p1].
    params : ParameterSet

    Outputs (dict of arrays)
    ------------------------
    - pLV : left ventricular pressure [mmHg]
    - Q2  : peripheral arterial flow [mL/s]
    - p1  : aortic/arterial pressure [mmHg]
    - Clv : ventricular compliance C_LV(t) [mL/mmHg]
    - dClv_dt : dC_LV/dt [mL/(mmHg*s)]
    - Elv : ventricular elastance E_LV(t)=1/C_LV(t) [mmHg/mL]
    - P0  : mitral inflow [mL/s]
    - P1  : aortic outflow [mL/s]
    - Vlv : left ventricular volume [mL] reconstructed as: Vlv = Vr + Clv*pLV
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float)

    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("x must have shape (N, 3) with columns [pLV, Q2, p1].")
    if t.shape[0] != x.shape[0]:
        raise ValueError("t and x must have the same length (N).")

    pLV = x[:, 0]
    Q2 = x[:, 1]
    p1 = x[:, 2]

    # Compliance signals (vectorized)
    Clv = clv(t, params)
    dClv = dclv_dt(t, params)
    Elv = elv(t, params)

    # Valve flows (vectorized)
    k = float(params.k_valve)
    P0 = mitral_flow(params.pLA, pLV, params.RMV, k)
    P1 = aortic_flow(pLV, p1, params.RAV, k)

    # Volume reconstruction
    Vlv = params.Vr + np.asarray(Clv, dtype=float) * pLV

    return {
        "pLV": np.asarray(pLV, dtype=float),
        "Q2": np.asarray(Q2, dtype=float),
        "p1": np.asarray(p1, dtype=float),
        "Clv": np.asarray(Clv, dtype=float),
        "dClv_dt": np.asarray(dClv, dtype=float),
        "Elv": np.asarray(Elv, dtype=float),
        "P0": np.asarray(P0, dtype=float),
        "P1": np.asarray(P1, dtype=float),
        "Vlv": np.asarray(Vlv, dtype=float),
    }
