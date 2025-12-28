import numpy as np

from cardio.physiology.activation import ecc, decc_dt, tvc_tvr


def test_activation_bounds():
    Tcc = 0.8
    Tvc, Tvr = tvc_tvr(Tcc)

    # Work in cycle time (tau) since ecc expects tau in [0, Tcc)
    tau = np.linspace(0.0, Tcc, 2000)
    y = ecc(tau, Tvc=Tvc, Tvr=Tvr, Tcc=Tcc)

    assert np.all(np.isfinite(y))
    assert np.min(y) >= -1e-9
    assert np.max(y) <= 1.0 + 1e-9


def test_activation_periodicity_via_modulo():
    Tcc = 0.8
    Tvc, Tvr = tvc_tvr(Tcc)

    rng = np.random.default_rng(0)
    t_abs = rng.uniform(0.0, 10.0, size=50)

    for t in t_abs:
        tau0 = np.mod(t, Tcc)
        tau1 = np.mod(t + Tcc, Tcc)
        tau2 = np.mod(t + 2 * Tcc, Tcc)

        y0 = float(ecc(tau0, Tvc=Tvc, Tvr=Tvr, Tcc=Tcc))
        y1 = float(ecc(tau1, Tvc=Tvc, Tvr=Tvr, Tcc=Tcc))
        y2 = float(ecc(tau2, Tvc=Tvc, Tvr=Tvr, Tcc=Tcc))

        assert np.isfinite(y0)
        assert abs(y0 - y1) < 1e-12
        assert abs(y0 - y2) < 1e-12


def test_activation_derivative_finite_and_consistent():
    Tcc = 0.8
    Tvc, Tvr = tvc_tvr(Tcc)

    eps = 1e-6
    # Choose safe taus away from boundaries: 0, Tvc, Tvc+Tvr, Tcc
    safe_taus = np.array([0.05, 0.15, 0.35, 0.55]) * Tcc

    for tau in safe_taus:
        d_analytical = float(decc_dt(tau, Tvc=Tvc, Tvr=Tvr, Tcc=Tcc))
        assert np.isfinite(d_analytical)

        y_plus = float(ecc(tau + eps, Tvc=Tvc, Tvr=Tvr, Tcc=Tcc))
        y_minus = float(ecc(tau - eps, Tvc=Tvc, Tvr=Tvr, Tcc=Tcc))
        d_num = (y_plus - y_minus) / (2 * eps)

        assert np.isfinite(d_num)
        # Loose sanity tolerance
        assert abs(d_analytical - d_num) < 1e-2


def test_activation_rest_phase_is_zero():
    Tcc = 0.8
    Tvc, Tvr = tvc_tvr(Tcc)

    tau_rest = np.linspace((Tvc + Tvr) + 1e-6, Tcc - 1e-6, 200)
    y = ecc(tau_rest, Tvc=Tvc, Tvr=Tvr, Tcc=Tcc)

    assert np.all(np.isfinite(y))
    assert np.max(np.abs(y)) < 1e-12
