from __future__ import annotations

import matplotlib.pyplot as plt

from cardio.analysis.metrics import compute_all_metrics
from cardio.params.dataclasses import SimulationConfig
from cardio.params.healthy import healthy_params
from cardio.params.pathology import (
    combined_stiffness_and_afterload,
    increased_afterload,
    reduced_arterial_compliance,
    arterial_stiffening_combo,
    
)
from cardio.plotting.plots import plot_clv, plot_p1, plot_pv_loop, plot_q2, plot_valve_flows
from cardio.simulation.pipeline import run_scenario_pair


def _print_comparison(title: str, mh: dict, mp: dict, keys: list[str]) -> None:
    print(f"\n=== {title} ===")
    for k in keys:
        hv = mh.get(k, None)
        pv = mp.get(k, None)
        if hv is None or pv is None:
            continue
        delta = pv - hv
        # avoid division by ~0
        rel = (delta / hv * 100.0) if abs(hv) > 1e-12 else float("nan")
        print(f"{k:18s}  healthy={hv:9.3f}   path={pv:9.3f}   Δ={delta:9.3f}   ({rel:6.1f}%)")


def main() -> None:
    # --- baseline ---
    healthy = healthy_params(label="healthy")

    # --- choose pathology scenario here ---
    # 1) arterial stiffening only
    # path = reduced_arterial_compliance(healthy, factor=0.5, label="Cart x0.5")
    #
    # 2) increased afterload only
    # path = increased_afterload(healthy, factor=1.5, label="Rart x1.5")
    #
    # 3) combined (recommended for a clear contrast)

    # path = combined_stiffness_and_afterload(
    #     healthy,
    #     compliance_factor=0.5,   # Cart reduced
    #     resistance_factor=1.5,   # Rart increased (=> Rtot increases too)
    #     label="arterial stiffening (Cart↓, R↑)",
    # )

    path = arterial_stiffening_combo(
    healthy,
    compliance_factor=0.3,
    resistance_factor=1.5,     # <-- clé : aucune augmentation de R
    inertance_factor=1.2,      # <-- optionnel
    label="Hypertension with arterial stiffening",
)


    # --- simulation config ---
    config = SimulationConfig(
        n_cycles=10,
        points_per_cycle=800,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
        enable_steady_state_check=False,  # (not implemented yet)
    )

    # --- run pair (warm-start pathology from healthy final state) ---
    res_h, res_p = run_scenario_pair(healthy=healthy, pathological=path, config=config)

    # --- compute metrics on last cycle ---
    mh = compute_all_metrics(res_h.signals, res_h.t, res_h.params.Tcc, valve_threshold=0.01)
    mp = compute_all_metrics(res_p.signals, res_p.t, res_p.params.Tcc, valve_threshold=0.01)

    print("\nHealthy label   :", res_h.params.label)
    print("Pathology label :", res_p.params.label)
    print("Pathology params:", f"Cart={res_p.params.Cart:.3g}, Rart={res_p.params.Rart:.3g}, Rtot={res_p.params.Rtot:.3g}")

    keys_main = [
        "p1_SBP", "p1_DBP", "p1_PP", "p1_MAP",
        "SV", "pLV_max",
        "Q2_mean", "Q2_max",
        "MV_open_fraction", "AV_open_fraction",
        "MV_peak", "AV_peak",
    ]
    _print_comparison("Key metrics (last cycle)", mh, mp, keys_main)

    # --- plots ---
    # Compliance (full time series is fine)
    plot_clv(res_h, show_last_cycle=False)
    #plot_clv(res_h,pathology=res_p, show_last_cycle=False)


    # 4 plots on last cycle for interpretation
    plot_p1(res_h, pathology=res_p, last_cycle=True)
    plot_pv_loop(res_h, pathology=res_p, last_cycle=True)
    plot_valve_flows(res_h, pathology=res_p, last_cycle=True)
    plot_q2(res_h, pathology=res_p, last_cycle=True)

    plt.show()


if __name__ == "__main__":
    main()
