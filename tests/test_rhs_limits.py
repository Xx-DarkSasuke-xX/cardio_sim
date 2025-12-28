import numpy as np

from cardio.models.systemic_nonlinear import rhs
from cardio.params.dataclasses import SimulationConfig
from cardio.params.healthy import healthy_params
from cardio.params.pathology import combined_stiffness_and_afterload
from cardio.simulation.pipeline import run_simulation


def test_rhs_shape_and_finite_on_reasonable_state():
    params = healthy_params()
    # state order: [pLV, Q2, p1]
    x = np.array([10.0, 50.0, 80.0], dtype=float)
    dx = rhs(t=0.1, x=x, params=params)

    assert isinstance(dx, np.ndarray)
    assert dx.shape == (3,)
    assert np.all(np.isfinite(dx))


def test_rhs_does_not_nan_for_small_times_over_cycle_grid():
    params = healthy_params()
    Tcc = params.Tcc

    # test a small set of states across the cycle
    times = np.linspace(0.0, Tcc, 50)
    x = np.array([params.pLA, 0.0, 80.0], dtype=float)

    for t in times:
        dx = rhs(t=float(t), x=x, params=params)
        assert dx.shape == (3,)
        assert np.all(np.isfinite(dx))


def test_short_simulation_is_finite_healthy():
    params = healthy_params()
    config = SimulationConfig(n_cycles=2, points_per_cycle=200, rtol=1e-6, atol=1e-8)

    res = run_simulation(params=params, config=config)
    assert res.x.shape[1] == 3
    assert np.all(np.isfinite(res.x))
    assert np.all(np.isfinite(res.signals["p1"]))
    assert np.all(np.isfinite(res.signals["Vlv"]))


def test_short_simulation_is_finite_pathology():
    healthy = healthy_params(label="healthy")
    path = combined_stiffness_and_afterload(
        healthy,
        compliance_factor=0.5,
        resistance_factor=1.5,
        label="arterial stiffening (Cart↓, R↑)",
    )
    config = SimulationConfig(n_cycles=2, points_per_cycle=200, rtol=1e-6, atol=1e-8)

    res = run_simulation(params=path, config=config)
    assert res.x.shape[1] == 3
    assert np.all(np.isfinite(res.x))
    assert np.all(np.isfinite(res.signals["p1"]))
    assert np.all(np.isfinite(res.signals["Vlv"]))
