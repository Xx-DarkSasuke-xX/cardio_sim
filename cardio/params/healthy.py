from __future__ import annotations

from cardio.params.dataclasses import ParameterSet


def healthy_params(label: str = "healthy") -> ParameterSet:
    """
    Baseline (healthy) parameter set used for numerical simulations.

    Values taken from the report table:
      - Tcc = 0.8 s (≈ 75 bpm)
      - Cmax = 15 mL/mmHg
      - Cmin = 0.4 mL/mmHg
      - Cart = 2 mL/mmHg
      - Iart = 1e-4 mmHg*s^2/mL
      - pLA = 8 mmHg
      - pRA = 3 mmHg
      - Rart = 0.1 mmHg*s/mL
      -Racp = 1 mmHg*s/mL
      - RAV = 0.1 mmHg*s/mL
      - RMV = 1e-2 mmHg*s/mL
      - Vr = 5 mL

    Notes:
      - The total peripheral resistance used in the model is derived as:
          Rtot =  Rart + Rcap
        
      - k_valve is a numerical hyperparameter controlling valve smoothing.
    """
    return ParameterSet(
        Tcc=0.8,
        Cmax=15.0,
        Cmin=0.4,
        pLA=8.0,
        pRA=3.0,
        RMV=1e-2,
        RAV=0.1,
        Cart=2.0,
        Iart=1e-4,
        Rart=0.1,
        Rcap = 1,
        Vr=5.0,
        k_valve=50.0,
        label=label,
        meta={},
    )
