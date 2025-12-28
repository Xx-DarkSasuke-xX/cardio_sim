from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np


ArrayLike = np.ndarray


@dataclass(frozen=True)
class ParameterSet:
    """
    Numerical parameters for the 0D systemic circulation model.

    Units (consistent with the report):
      - Pressures: mmHg
      - Volumes: mL
      - Flows: mL/s
      - Time: s
      - Resistances: mmHg*s/mL
      - Compliance: mL/mmHg
      - Inertance: mmHg*s^2/mL

    Notes:
      - Provisional rule (per project decision): Rcap = Rart => Rtot = 2*Rart.
        This is encoded via the derived property `Rtot`.
    """

    # --- Cardiac cycle timing ---
    Tcc: float  # cardiac cycle duration [s]

    # --- Ventricular compliance envelope ---
    Cmax: float  # maximal ventricular compliance [mL/mmHg]
    Cmin: float  # minimal ventricular compliance [mL/mmHg]

    # --- Atrial / venous pressures ---
    pLA: float  # left atrial pressure [mmHg]
    pRA: float  # right atrial pressure (venous) [mmHg]

    # --- Valve resistances ---
    RMV: float  # mitral valve resistance [mmHg*s/mL]
    RAV: float  # aortic valve resistance [mmHg*s/mL]

    # --- Arterial (Windkessel) parameters ---
    Cart: float  # arterial compliance [mL/mmHg]
    Iart: float  # arterial inertance [mmHg*s^2/mL]
    Rart: float  # arterial resistance [mmHg*s/mL] (used as building block)
    Rcap: float  # capillary resistance [mmHg*s/mL]

    # --- Ventricular residual volume (used for Vlv reconstruction) ---
    Vr: float  # residual ventricular volume [mL]

    # --- Numerical / smoothing hyperparameters ---
    k_valve: float = 50.0  # slope for smooth Heaviside approximation (tanh)

    # --- Optional metadata ---
    label: str = "healthy"
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def Rtot(self) -> float:
        """
        Total peripheral resistance used in Eq.57.

        Provisional assumption:
          Rcap = Rart  =>  Rtot = Rcap + Rart = 2 * Rart
        """
        return 2.0 * self.Rart


@dataclass(frozen=True)
class SimulationConfig:
    """
    Simulation configuration (independent of physiology).

    This controls how we run simulations (time span, sampling, solver settings),
    without embedding any model equations.
    """

    n_cycles: int = 10
    points_per_cycle: int = 800

    # solver selection (to be used later in integrate.py)
    method: str = "RK45"

    # tolerances (used later when solve_ivp is implemented)
    rtol: float = 1e-6
    atol: float = 1e-8

    # whether to stop early if periodic steady-state is detected
    enable_steady_state_check: bool = True
    steady_state_tol: float = 1e-3  # tolerance on cycle-to-cycle difference


@dataclass(frozen=True)
class SimulationResult:
    """
    Container for simulation outputs.

    Required:
      - t: time vector
      - x: state matrix with columns [pLV, Q2, p1]
      - params: ParameterSet used
      - config: SimulationConfig used

    Optional derived signals:
      - signals: dict for reconstructed variables (e.g., Vlv, P0, P1, Clv, dClv_dt)
      - metrics: dict for summary metrics (SBP/DBP/PP/MAP, SV, valve timing, ...)
    """

    t: ArrayLike
    x: ArrayLike

    params: ParameterSet
    config: SimulationConfig

    signals: Dict[str, ArrayLike] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)

    def state_names(self) -> Sequence[str]:
        return ("pLV", "Q2", "p1")

    def get_state(self, name: str) -> ArrayLike:
        """
        Convenience accessor: result.get_state("p1") -> array.
        """
        idx = {n: i for i, n in enumerate(self.state_names())}.get(name)
        if idx is None:
            raise KeyError(f"Unknown state '{name}'. Valid: {self.state_names()}")
        return self.x[:, idx]


@dataclass(frozen=True)
class Scenario:
    """
    A scenario is a named transformation applied to a base ParameterSet.

    Example: start from healthy_params(), apply reduced_compliance() to get pathology.
    """
    name: str
    transform: Callable[[ParameterSet], ParameterSet]
    description: str = ""


def make_time_grid(Tcc: float, n_cycles: int, points_per_cycle: int) -> np.ndarray:
    """
    Build a uniform time grid spanning n_cycles of duration Tcc.

    This helper is intentionally simple and used across the project to ensure
    consistent sampling.
    """
    t_end = float(n_cycles) * float(Tcc)
    n_points = int(n_cycles) * int(points_per_cycle) + 1
    return np.linspace(0.0, t_end, n_points, dtype=float)
