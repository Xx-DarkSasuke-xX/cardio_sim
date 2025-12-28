from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.integrate import solve_ivp

from cardio.models.systemic_nonlinear import rhs
from cardio.params.dataclasses import ParameterSet, SimulationConfig


def integrate_system(
    params: ParameterSet,
    config: SimulationConfig,
    x0: np.ndarray,
    t_eval: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrate the nonlinear ODE system over the provided time grid.

    Parameters
    ----------
    params : ParameterSet
        Model parameters.
    config : SimulationConfig
        Numerical configuration (solver method, tolerances).
    x0 : np.ndarray
        Initial state vector [pLV0, Q2_0, p1_0].
    t_eval : np.ndarray
        Time points where the solution is sampled (monotonic increasing).

    Returns
    -------
    t : (N,) np.ndarray
        Time vector (same as t_eval if solver succeeds).
    x : (N,3) np.ndarray
        State trajectory with columns [pLV, Q2, p1].
    """
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    if x0.size != 3:
        raise ValueError("x0 must have size 3: [pLV0, Q2_0, p1_0].")

    t_eval = np.asarray(t_eval, dtype=float).reshape(-1)
    if t_eval.size < 2:
        raise ValueError("t_eval must contain at least 2 time points.")
    if not np.all(np.diff(t_eval) > 0):
        raise ValueError("t_eval must be strictly increasing.")

    t0 = float(t_eval[0])
    tf = float(t_eval[-1])

    # Wrapper to match solve_ivp signature
    def f(t: float, x: np.ndarray) -> np.ndarray:
        return rhs(t, x, params)

    sol = solve_ivp(
        fun=f,
        t_span=(t0, tf),
        y0=x0,
        method=config.method,
        t_eval=t_eval,
        rtol=config.rtol,
        atol=config.atol,
        vectorized=False,
    )

    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    # sol.y has shape (n_states, N). We return (N, n_states).
    t = np.asarray(sol.t, dtype=float)
    x = np.asarray(sol.y.T, dtype=float)

    if x.shape[1] != 3:
        raise RuntimeError(f"Unexpected state dimension returned by solver: {x.shape}")

    return t, x
