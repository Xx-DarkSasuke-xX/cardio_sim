from __future__ import annotations

import matplotlib.pyplot as plt

from cardio.analysis.metrics import compute_all_metrics
from cardio.params.dataclasses import SimulationConfig
from cardio.params.healthy import healthy_params
from cardio.plotting.plots import plot_clv, plot_p1, plot_pv_loop, plot_q2, plot_valve_flows
from cardio.simulation.pipeline import run_simulation


def main() -> None:
    # --- Parameters & simulation config ---
    params = healthy_params()

    # You can increase n_cycles to help reach periodic steady state
    config = SimulationConfig(
        n_cycles=10,
        points_per_cycle=800,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
        enable_steady_state_check=False,  # (not implemented yet)
    )

    # --- Run simulation ---
    res = run_simulation(params=params, config=config)

    # --- Metrics on the last cycle ---
    metrics = compute_all_metrics(
        signals=res.signals,
        t=res.t,
        Tcc=res.params.Tcc,
        valve_threshold=0.01,
    )
    res = res.__class__(**{**res.__dict__, "metrics": metrics})  # keep dataclass frozen-safe

    print("\n=== Healthy simulation (last cycle metrics) ===")
    print(f"p1_SBP  : {metrics['p1_SBP']:.2f} mmHg")
    print(f"p1_DBP  : {metrics['p1_DBP']:.2f} mmHg")
    print(f"p1_PP   : {metrics['p1_PP']:.2f} mmHg")
    print(f"p1_MAP  : {metrics['p1_MAP']:.2f} mmHg")
    print(f"SV      : {metrics['SV']:.2f} mL")
    print(f"pLV_max : {metrics['pLV_max']:.2f} mmHg")
    print(f"Q2_mean : {metrics['Q2_mean']:.2f} mL/s")
    print(f"MV open fraction: {metrics['MV_open_fraction']:.3f}")
    print(f"AV open fraction: {metrics['AV_open_fraction']:.3f}")

    # --- Plots ---
    plot_clv(res, show_last_cycle=False)
    plot_p1(res, pathology=None, last_cycle=True)
    plot_pv_loop(res, pathology=None, last_cycle=True)
    plot_valve_flows(res, pathology=None, last_cycle=True)
    plot_q2(res, pathology=None, last_cycle=True)

    plt.show()


if __name__ == "__main__":
    main()
