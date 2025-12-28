from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from cardio.analysis.metrics import compute_all_metrics
from cardio.config.defaults import get_default_config, get_default_healthy_params
from cardio.params.pathology import combined_stiffness_and_afterload
from cardio.plotting.plots import plot_clv, plot_p1, plot_pv_loop, plot_q2, plot_valve_flows
from cardio.simulation.pipeline import run_scenario_pair


def _save(fig, outdir: Path, name: str, dpi: int = 300) -> None:
    out_png = outdir / f"{name}.png"
    out_pdf = outdir / f"{name}.pdf"
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")


def main() -> None:
    outdir = Path("exports") / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    config = get_default_config()
    healthy = get_default_healthy_params()

    pathology = combined_stiffness_and_afterload(
        healthy,
        compliance_factor=0.5,
        resistance_factor=1.5,
        label="Hypertension with arterial stiffening",
    )

    res_h, res_p = run_scenario_pair(healthy=healthy, pathological=pathology, config=config)

    # Compute metrics (optional export later)
    mh = compute_all_metrics(res_h.signals, res_h.t, res_h.params.Tcc)
    mp = compute_all_metrics(res_p.signals, res_p.t, res_p.params.Tcc)

    # Simple console summary (kept short)
    print("\nExporting figures to:", outdir.resolve())
    print("Healthy p1 SBP/DBP:", f"{mh['p1_SBP']:.2f}/{mh['p1_DBP']:.2f} mmHg")
    print("Path   p1 SBP/DBP:", f"{mp['p1_SBP']:.2f}/{mp['p1_DBP']:.2f} mmHg")

    # 1) Compliance over full simulation
    fig1, ax1 = plt.subplots()
    plot_clv(res_h, show_last_cycle=False, ax=ax1)
    # Pathology has identical LV compliance in this scenario (same Cmin/Cmax), so we don't overlay.
    _save(fig1, outdir, "01_clv_time")
    plt.close(fig1)

    # 2) Arterial pressure (last cycle)
    fig2, ax2 = plt.subplots()
    plot_p1(res_h, pathology=res_p, last_cycle=True, ax=ax2)
    _save(fig2, outdir, "02_p1_last_cycle_compare")
    plt.close(fig2)

    # 3) PV loop (last cycle)
    fig3, ax3 = plt.subplots()
    plot_pv_loop(res_h, pathology=res_p, last_cycle=True, ax=ax3)
    _save(fig3, outdir, "03_pv_loop_last_cycle_compare")
    plt.close(fig3)

    # 4) Valve flows (last cycle)
    fig4, ax4 = plt.subplots()
    plot_valve_flows(res_h, pathology=res_p, last_cycle=True, ax=ax4)
    _save(fig4, outdir, "04_valve_flows_last_cycle_compare")
    plt.close(fig4)

    # 5) Peripheral flow Q2 (last cycle)
    fig5, ax5 = plt.subplots()
    plot_q2(res_h, pathology=res_p, last_cycle=True, ax=ax5)
    _save(fig5, outdir, "05_q2_last_cycle_compare")
    plt.close(fig5)

    print("Done.")


if __name__ == "__main__":
    main()
