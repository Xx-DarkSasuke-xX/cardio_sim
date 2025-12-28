from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from cardio.models.signals import reconstruct_signals
from cardio.params.dataclasses import ParameterSet, SimulationConfig, SimulationResult, make_time_grid
from cardio.simulation.initial_conditions import default_initial_state

# integrate_system will be implemented in cardio/simulation/integrate.py
from cardio.simulation.integrate import integrate_system


def run_simulation(
    params: ParameterSet,
    config: SimulationConfig,
    x0: Optional[np.ndarray] = None,
) -> SimulationResult:
    """
    Run a forward nonlinear simulation over multiple cardiac cycles.

    This function:
      1) builds a time grid
      2) integrates the ODE system
      3) reconstructs derived signals needed for analysis/plots
      4) returns a SimulationResult container

    No metrics or plotting are computed here (those belong to analysis/plotting modules).
    """
    if x0 is None:
        x0 = default_initial_state(params)

    t_eval = make_time_grid(params.Tcc, config.n_cycles, config.points_per_cycle)

    # Integrate ODEs (Eq.56–58 through models.systemic_nonlinear.rhs)
    t, x = integrate_system(params=params, config=config, x0=x0, t_eval=t_eval)

    # Reconstruct derived signals (Vlv, valve flows, compliance, ...)
    signals = reconstruct_signals(t=t, x=x, params=params)

    return SimulationResult(
        t=t,
        x=x,
        params=params,
        config=config,
        signals=signals,
        metrics={},
        notes={},
    )


def run_scenario_pair(
    healthy: ParameterSet,
    pathological: ParameterSet,
    config: SimulationConfig,
    x0: Optional[np.ndarray] = None,
) -> tuple[SimulationResult, SimulationResult]:
    """
    Convenience helper: run healthy and pathological simulations using the same config.

    This is mainly used by scripts (compare plots/metrics).
    """
    res_h = run_simulation(healthy, config=config, x0=x0)
    # Use final state of healthy as warm-start for pathological if x0 not provided
    x0_path = res_h.x[-1, :] if x0 is None else x0
    res_p = run_simulation(pathological, config=config, x0=x0_path)
    return res_h, res_p
