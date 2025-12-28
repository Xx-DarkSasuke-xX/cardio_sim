import numpy as np

from cardio.params.healthy import healthy_params
from cardio.physiology.compliance import clv, dclv_dt, elv


def test_clv_bounds_over_multiple_cycles():
    params = healthy_params()
    Tcc = params.Tcc

    t = np.linspace(0.0, 5.0 * Tcc, 5000)
    C = clv(t, params)

    assert np.all(np.isfinite(C))
    assert np.min(C) >= params.Cmin - 1e-9
    assert np.max(C) <= params.Cmax + 1e-9


def test_dclv_dt_is_finite():
    params = healthy_params()
    Tcc = params.Tcc

    t = np.linspace(0.0, 3.0 * Tcc, 3000)
    dC = dclv_dt(t, params)

    assert np.all(np.isfinite(dC))


def test_dclv_dt_matches_finite_difference_sanity():
    params = healthy_params()
    Tcc = params.Tcc

    # central difference check at "safe" times (avoid boundaries)
    eps = 1e-6
    safe_times = np.array([0.05, 0.15, 0.35, 0.55]) * Tcc

    for t0 in safe_times:
        dC_ana = float(dclv_dt(t0, params))
        assert np.isfinite(dC_ana)

        C_plus = float(clv(t0 + eps, params))
        C_minus = float(clv(t0 - eps, params))
        dC_num = (C_plus - C_minus) / (2 * eps)

        assert np.isfinite(dC_num)
        # loose tolerance: we just want a sanity check
        assert abs(dC_ana - dC_num) < 1e-2


def test_elv_is_inverse_of_clv():
    params = healthy_params()
    Tcc = params.Tcc

    t = np.linspace(0.0, 2.0 * Tcc, 2000)
    C = clv(t, params)
    E = elv(t, params)

    assert np.all(np.isfinite(C))
    assert np.all(np.isfinite(E))

    # E*C should be ~1
    prod = np.asarray(C) * np.asarray(E)
    assert np.max(np.abs(prod - 1.0)) < 1e-9
