from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from cardio.params.healthy import healthy_params
# from cardio.params.pathology import 
from cardio.params.pathology import arterial_stiffening_combo

from cardio.analysis.linearization import (
    build_arterial_lti,
    arterial_poles_zeros_from_tf,
)
from cardio.plotting.lti_plots import plot_pole_zero_map


def _save(fig: plt.Figure, outdir: Path, name: str, dpi: int = 300) -> None:
    out_png = outdir / f"{name}.png"
    out_pdf = outdir / f"{name}.pdf"
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight", format="pdf")


def _impulse(sys_tf: signal.TransferFunction, t: np.ndarray) -> np.ndarray:
    _, y = signal.impulse(sys_tf, T=t)
    return np.asarray(y, dtype=float)


def _step(sys_tf: signal.TransferFunction, t: np.ndarray) -> np.ndarray:
    _, y = signal.step(sys_tf, T=t)
    return np.asarray(y, dtype=float)


def _bode(sys_tf: signal.TransferFunction, w: np.ndarray):
    _, H = signal.freqresp(sys_tf, w=w)
    H = np.asarray(H)
    mag = np.abs(H)
    phase_deg = np.angle(H, deg=True)
    return mag, phase_deg


def main() -> None:
    outdir = Path("exports") / "figures_lti"
    outdir.mkdir(parents=True, exist_ok=True)

    # Params
    p_h = healthy_params()
    p_p = arterial_stiffening_combo(p_h)

    lti_h = build_arterial_lti(p_h)
    lti_p = build_arterial_lti(p_p)

    poles_h, zeros_h = arterial_poles_zeros_from_tf(lti_h.a0, lti_h.a1, lti_h.b0, lti_h.b1)
    poles_p, zeros_p = arterial_poles_zeros_from_tf(lti_p.a0, lti_p.a1, lti_p.b0, lti_p.b1)

    print("Exporting LTI figures to:", outdir.resolve())

    # ------------------------------------------------------------------
    # 1) Pole-zero map (overlay)
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots()
    plot_pole_zero_map(poles_h, zeros_h, title="Arterial Windkessel LTI — Pole-zero map", ax=ax1)
    # overlay pathology with different markers/colors without styling overload
    ax1.scatter(np.real(zeros_p), np.imag(zeros_p), marker="o", facecolors="none", edgecolors="tab:blue", label="zeros (Hypertension with arterial stiffening)")
    ax1.scatter(np.real(poles_p), np.imag(poles_p), marker="x", color="tab:blue", label="poles (Hypertension with arterial stiffening)")
    ax1.legend(loc="best")
    _save(fig1, outdir, "lti_01_pole_zero_map_compare")
    plt.close(fig1)

    # Common time grid for time responses
    t_end = 5.0
    t = np.linspace(0.0, t_end, 2000)

    # ------------------------------------------------------------------
    # 2) Impulse response (overlay)
    # ------------------------------------------------------------------
    y_imp_h = _impulse(lti_h.sys_tf, t)
    y_imp_p = _impulse(lti_p.sys_tf, t)

    fig2, ax2 = plt.subplots()
    ax2.plot(t, y_imp_h, label="healthy")
    ax2.plot(t, y_imp_p, label="Hypertension with arterial stiffening")
    ax2.set_title("Impulse response: Δp1 / ΔQin")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Δp1 (arb. units)")
    ax2.grid(True, linestyle="--", linewidth=0.5)
    ax2.legend(loc="best")
    _save(fig2, outdir, "lti_02_impulse_response_compare")
    plt.close(fig2)

    # ------------------------------------------------------------------
    # 3) Step response (overlay)
    # ------------------------------------------------------------------
    y_step_h = _step(lti_h.sys_tf, t)
    y_step_p = _step(lti_p.sys_tf, t)

    fig3, ax3 = plt.subplots()
    ax3.plot(t, y_step_h, label="healthy")
    ax3.plot(t, y_step_p, label="Hypertension with arterial stiffening")
    ax3.set_title("Step response: Δp1 / ΔQin")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Δp1 (arb. units)")
    ax3.grid(True, linestyle="--", linewidth=0.5)
    ax3.legend(loc="best")
    _save(fig3, outdir, "lti_03_step_response_compare")
    plt.close(fig3)

    # ------------------------------------------------------------------
    # 4) Bode (magnitude + phase) — overlay
    # ------------------------------------------------------------------
    w = np.logspace(-2, 3, 1200)  # rad/s
    mag_h, ph_h = _bode(lti_h.sys_tf, w)
    mag_p, ph_p = _bode(lti_p.sys_tf, w)

    fig4, (ax4a, ax4b) = plt.subplots(2, 1, sharex=True)
    fig4.suptitle("Bode plot: Δp1 / ΔQin")

    ax4a.semilogx(w, 20.0 * np.log10(np.maximum(mag_h, 1e-30)), label="healthy")
    ax4a.semilogx(w, 20.0 * np.log10(np.maximum(mag_p, 1e-30)), label="Hypertension with arterial stiffening")
    ax4a.set_ylabel("Magnitude (dB)")
    ax4a.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax4a.legend(loc="best")

    ax4b.semilogx(w, ph_h, label="healthy")
    ax4b.semilogx(w, ph_p, label="Hypertension with arterial stiffening")
    ax4b.set_ylabel("Phase (deg)")
    ax4b.set_xlabel("Frequency ω (rad/s)")
    ax4b.grid(True, which="both", linestyle="--", linewidth=0.5)

    _save(fig4, outdir, "lti_04_bode_compare")
    plt.close(fig4)

    print("Done.")


if __name__ == "__main__":
    main()
