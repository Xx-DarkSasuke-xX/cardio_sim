"""
Run LTI analysis for the linearized arterial Windkessel sub-model.

This script:
- builds the arterial LTI (A,B,C,D) for healthy and arterial stiffening
- checks observability numerically
- computes poles, zeros, natural frequency, damping
- verifies structural identifiability (roundtrip)
- produces standard LTI plots (pole-zero, impulse, step, bode)
"""

import numpy as np
import matplotlib.pyplot as plt

from cardio.params.healthy import healthy_params
from cardio.params.pathology import combined_stiffness_and_afterload
# from cardio.params.pathology import arterial_stiffening_combo


from cardio.analysis.linearization import (
    build_arterial_lti,
    arterial_poles_zeros_from_tf,
    arterial_frequency_params,
    arterial_expected_zero,
)
from cardio.analysis.observability import (
    observability_checks,
    observability_gramian_eigs,
)
from cardio.analysis.identifiability import roundtrip_identifiability
from cardio.plotting.lti_plots import plot_lti_summary


def print_header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def analyze_case(label: str, params):
    print_header(f"LTI ANALYSIS — {label}")

    # Build LTI
    lti = build_arterial_lti(params)

    # --- Observability ---
    obs = observability_checks(lti.A, lti.C)
    print("Observability:")
    print(f"  rank(O) = {obs.rank} (n=2)")
    print(f"  det(O)  = {obs.det}")
    print(f"  cond(O) = {obs.cond:.3e}")

    # Gramian (optional but informative)
    try:
        _, eigs_Wo = observability_gramian_eigs(lti.A, lti.C)
        print("  Gramian eigenvalues:", eigs_Wo)
    except Exception as e:
        print("  Gramian not computed:", e)

    # --- Poles / Zeros ---
    poles, zeros = arterial_poles_zeros_from_tf(lti.a0, lti.a1, lti.b0, lti.b1)
    wn, zeta = arterial_frequency_params(lti.a0, lti.a1)
    s0_expected = arterial_expected_zero(lti.b0, lti.b1)

    print("\nDynamics:")
    print("  Poles :", poles)
    print("  Zeros :", zeros)
    print(f"  Expected zero s0 = {s0_expected:.4f}")
    print(f"  Natural frequency ω_n = {wn:.4f} rad/s")
    print(f"  Damping ratio ζ = {zeta:.4f}")

    # --- Identifiability (roundtrip) ---
    ident = roundtrip_identifiability(params)
    print("\nIdentifiability (roundtrip):")
    print(f"  C_hat = {ident.C_hat:.6g} | rel err = {ident.rel_err_C:.3e}")
    print(f"  I_hat = {ident.I_hat:.6g} | rel err = {ident.rel_err_I:.3e}")
    print(f"  R_hat = {ident.R_hat:.6g} | rel err = {ident.rel_err_R:.3e}")

    # --- Plots ---
    plot_lti_summary(
        poles=poles,
        zeros=zeros,
        sys_tf=lti.sys_tf,
        title_prefix=f"{label}",
        t_end=5.0,
    )


def main():
    # Healthy parameters
    params_h = healthy_params()

    # Pathology: arterial stiffening (Cart ↓, R ↑)
    params_p = combined_stiffness_and_afterload(params_h)
    # params_p =arterial_stiffening_combo(params_h)

    analyze_case("Healthy arterial system", params_h)
    analyze_case("Hypertension with arterial stiffening", params_p)

    plt.show()


if __name__ == "__main__":
    main()
