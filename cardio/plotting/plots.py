from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from cardio.params.dataclasses import SimulationResult


def _last_cycle_slice(res: SimulationResult) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (t, idx_mask) for the last cardiac cycle in a SimulationResult.
    """
    t = np.asarray(res.t, dtype=float).reshape(-1)
    Tcc = float(res.params.Tcc)
    t_end = t[-1]
    start = t_end - Tcc
    mask = t >= start
    return t, mask


def plot_clv(res: SimulationResult, show_last_cycle: bool = False, ax=None):
    """
    Plot ventricular compliance C_LV(t).
    """
    if "Clv" not in res.signals:
        raise KeyError("Signal 'Clv' not found in SimulationResult.signals")

    t = np.asarray(res.t, dtype=float)
    clv = np.asarray(res.signals["Clv"], dtype=float)

    if ax is None:
        fig, ax = plt.subplots()

    if show_last_cycle:
        _, mask = _last_cycle_slice(res)
        ax.plot(t[mask], clv[mask], label=f"{res.params.label} (last cycle)")
        ax.set_xlabel("Time [s] (last cycle)")
    else:
        ax.plot(t, clv, label=res.params.label)
        ax.set_xlabel("Time [s]")

    ax.set_ylabel("C_LV(t) [mL/mmHg]")
    ax.set_title("Left Ventricular Compliance")
    ax.grid(True)
    ax.legend()
    return ax


def plot_p1(
    healthy: SimulationResult,
    pathology: Optional[SimulationResult] = None,
    last_cycle: bool = True,
    ax=None,
):
    """
    Plot arterial/aortic pressure p1(t), healthy vs pathology.
    """
    if "p1" not in healthy.signals:
        raise KeyError("Signal 'p1' not found in healthy.signals")

    if ax is None:
        fig, ax = plt.subplots()

    if last_cycle:
        t_h, m_h = _last_cycle_slice(healthy)
        ax.plot(t_h[m_h], healthy.signals["p1"][m_h], label=f"{healthy.params.label}")
        ax.set_xlabel("Time [s] (last cycle)")
    else:
        ax.plot(healthy.t, healthy.signals["p1"], label=f"{healthy.params.label}")
        ax.set_xlabel("Time [s]")

    if pathology is not None:
        if "p1" not in pathology.signals:
            raise KeyError("Signal 'p1' not found in pathology.signals")
        if last_cycle:
            t_p, m_p = _last_cycle_slice(pathology)
            ax.plot(t_p[m_p], pathology.signals["p1"][m_p], label=f"{pathology.params.label}")
        else:
            ax.plot(pathology.t, pathology.signals["p1"], label=f"{pathology.params.label}")

    ax.set_ylabel("p1(t) [mmHg]")
    ax.set_title("Arterial (Aortic) Pressure")
    ax.grid(True)
    ax.legend()
    return ax


def plot_pv_loop(
    healthy: SimulationResult,
    pathology: Optional[SimulationResult] = None,
    last_cycle: bool = True,
    ax=None,
):
    """
    Plot PV loop: pLV(t) vs Vlv(t), healthy vs pathology.
    """
    for key in ("pLV", "Vlv"):
        if key not in healthy.signals:
            raise KeyError(f"Signal '{key}' not found in healthy.signals")

    if ax is None:
        fig, ax = plt.subplots()

    if last_cycle:
        _, m_h = _last_cycle_slice(healthy)
        ax.plot(
            healthy.signals["Vlv"][m_h],
            healthy.signals["pLV"][m_h],
            label=f"{healthy.params.label}",
        )
    else:
        ax.plot(healthy.signals["Vlv"], healthy.signals["pLV"], label=f"{healthy.params.label}")

    if pathology is not None:
        for key in ("pLV", "Vlv"):
            if key not in pathology.signals:
                raise KeyError(f"Signal '{key}' not found in pathology.signals")
        if last_cycle:
            _, m_p = _last_cycle_slice(pathology)
            ax.plot(
                pathology.signals["Vlv"][m_p],
                pathology.signals["pLV"][m_p],
                label=f"{pathology.params.label}",
            )
        else:
            ax.plot(pathology.signals["Vlv"], pathology.signals["pLV"], label=f"{pathology.params.label}")

    ax.set_xlabel("V_LV(t) [mL]")
    ax.set_ylabel("p_LV(t) [mmHg]")
    ax.set_title("Pressure–Volume Loop")
    ax.grid(True)
    ax.legend()
    return ax


def plot_valve_flows(
    healthy: SimulationResult,
    pathology: Optional[SimulationResult] = None,
    last_cycle: bool = True,
    ax=None,
):
    """
    Plot valve flows P0(t) (mitral) and P1(t) (aortic).
    """
    for key in ("P0", "P1"):
        if key not in healthy.signals:
            raise KeyError(f"Signal '{key}' not found in healthy.signals")

    if ax is None:
        fig, ax = plt.subplots()

    if last_cycle:
        t_h, m_h = _last_cycle_slice(healthy)
        ax.plot(t_h[m_h], healthy.signals["P0"][m_h], label=f"MV P0 — {healthy.params.label}")
        ax.plot(t_h[m_h], healthy.signals["P1"][m_h], label=f"AV P1 — {healthy.params.label}")
        ax.set_xlabel("Time [s] (last cycle)")
    else:
        ax.plot(healthy.t, healthy.signals["P0"], label=f"MV P0 — {healthy.params.label}")
        ax.plot(healthy.t, healthy.signals["P1"], label=f"AV P1 — {healthy.params.label}")
        ax.set_xlabel("Time [s]")

    if pathology is not None:
        for key in ("P0", "P1"):
            if key not in pathology.signals:
                raise KeyError(f"Signal '{key}' not found in pathology.signals")
        if last_cycle:
            t_p, m_p = _last_cycle_slice(pathology)
            ax.plot(t_p[m_p], pathology.signals["P0"][m_p], label=f"MV P0 — {pathology.params.label}")
            ax.plot(t_p[m_p], pathology.signals["P1"][m_p], label=f"AV P1 — {pathology.params.label}")
        else:
            ax.plot(pathology.t, pathology.signals["P0"], label=f"MV P0 — {pathology.params.label}")
            ax.plot(pathology.t, pathology.signals["P1"], label=f"AV P1 — {pathology.params.label}")

    ax.set_ylabel("Flow [mL/s]")
    ax.set_title("Valve Flows (Mitral and Aortic)")
    ax.grid(True)
    ax.legend()
    return ax


def plot_q2(
    healthy: SimulationResult,
    pathology: Optional[SimulationResult] = None,
    last_cycle: bool = True,
    ax=None,
):
    """
    Plot peripheral arterial flow Q2(t), healthy vs pathology.
    """
    if "Q2" not in healthy.signals:
        raise KeyError("Signal 'Q2' not found in healthy.signals")

    if ax is None:
        fig, ax = plt.subplots()

    if last_cycle:
        t_h, m_h = _last_cycle_slice(healthy)
        ax.plot(t_h[m_h], healthy.signals["Q2"][m_h], label=f"{healthy.params.label}")
        ax.set_xlabel("Time [s] (last cycle)")
    else:
        ax.plot(healthy.t, healthy.signals["Q2"], label=f"{healthy.params.label}")
        ax.set_xlabel("Time [s]")

    if pathology is not None:
        if "Q2" not in pathology.signals:
            raise KeyError("Signal 'Q2' not found in pathology.signals")
        if last_cycle:
            t_p, m_p = _last_cycle_slice(pathology)
            ax.plot(t_p[m_p], pathology.signals["Q2"][m_p], label=f"{pathology.params.label}")
        else:
            ax.plot(pathology.t, pathology.signals["Q2"], label=f"{pathology.params.label}")

    ax.set_ylabel("Q2(t) [mL/s]")
    ax.set_title("Peripheral Arterial Flow")
    ax.grid(True)
    ax.legend()
    return ax
