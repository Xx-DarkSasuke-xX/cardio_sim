from __future__ import annotations

import numpy as np

from cardio.params.dataclasses import ParameterSet


def default_initial_state(params: ParameterSet) -> np.ndarray:
    """
    Default initial state for the nonlinear systemic model.

    State order is fixed:
        x0 = [pLV0, Q2_0, p1_0]

    These values are chosen to be reasonable starting points that typically allow
    the solver to converge to a periodic steady-state after a few cycles.

    Notes
    -----
    - pLV0 is initialized near atrial pressure (diastolic filling starting point).
    - p1_0 is initialized near a typical diastolic arterial pressure.
    - Q2_0 is initialized to a small positive value.

    The project uses multi-cycle simulations; transient mismatch in the first cycle
    is expected and acceptable.
    """
    pLV0 = float(params.pLA)  # start close to atrial pressure
    p1_0 = 80.0               # typical diastolic aortic pressure (mmHg)
    Q2_0 = 0.0                # start with no flow (safe neutral)
    return np.array([pLV0, Q2_0, p1_0], dtype=float)


def initial_state_from_guess(pLV0: float, Q2_0: float, p1_0: float) -> np.ndarray:
    """
    Build a state vector from user-provided initial values.

    This helper keeps the state ordering consistent across the project.
    """
    return np.array([float(pLV0), float(Q2_0), float(p1_0)], dtype=float)
