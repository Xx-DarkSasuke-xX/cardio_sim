"""
Default configuration values for cardiovascular simulations.

This module centralizes numerical and simulation-level defaults
used across scripts (healthy, pathology, comparisons).

The goal is to ensure consistency and avoid duplicated parameters
in multiple entry points.
"""

from __future__ import annotations

from cardio.params.dataclasses import SimulationConfig
from cardio.params.healthy import healthy_params


# ---------------------------------------------------------------------
# Simulation defaults
# ---------------------------------------------------------------------

DEFAULT_SIMULATION_CONFIG = SimulationConfig(
    n_cycles=10,             # number of cardiac cycles to simulate
    points_per_cycle=800,    # temporal resolution per cycle
    method="RK45",           # ODE solver
    rtol=1e-6,               # relative tolerance
    atol=1e-8,               # absolute tolerance
    enable_steady_state_check=False,  # future extension
)


# ---------------------------------------------------------------------
# Baseline physiological parameters
# ---------------------------------------------------------------------

DEFAULT_HEALTHY_PARAMS = healthy_params(label="healthy")


# ---------------------------------------------------------------------
# Helper accessors (optional but convenient)
# ---------------------------------------------------------------------

def get_default_config() -> SimulationConfig:
    """Return a copy of the default simulation configuration."""
    return DEFAULT_SIMULATION_CONFIG


def get_default_healthy_params():
    """Return baseline healthy cardiovascular parameters."""
    return DEFAULT_HEALTHY_PARAMS
