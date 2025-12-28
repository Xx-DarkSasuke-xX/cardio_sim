from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


def _last_cycle_mask(t: np.ndarray, Tcc: float) -> np.ndarray:
    """
    Return a boolean mask selecting the last cardiac cycle in t.

    Assumes t starts at 0 and spans multiple cycles.
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    if t.size < 2:
        raise ValueError("t must have at least 2 samples.")
    if Tcc <= 0:
        raise ValueError("Tcc must be > 0.")

    t_end = t[-1]
    start = t_end - Tcc
    return t >= start


def _trapz_mean(y: np.ndarray, t: np.ndarray) -> float:
    """Time-average using trapezoidal integration."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    if y.size != t.size:
        raise ValueError("y and t must have same length.")
    duration = t[-1] - t[0]
    if duration <= 0:
        raise ValueError("Invalid time duration.")
    area = np.trapz(y, t)
    return float(area / duration)


def arterial_pressure_metrics(p1: np.ndarray, t: np.ndarray) -> Dict[str, float]:
    """
    Basic arterial pressure metrics computed on the provided segment.

    Returns:
      - SBP: systolic blood pressure (max)
      - DBP: diastolic blood pressure (min)
      - PP : pulse pressure (SBP-DBP)
      - MAP: mean arterial pressure (time average)
    """
    p1 = np.asarray(p1, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)
    if p1.size != t.size:
        raise ValueError("p1 and t must have same length.")

    sbp = float(np.max(p1))
    dbp = float(np.min(p1))
    pp = sbp - dbp
    map_ = _trapz_mean(p1, t)

    return {"SBP": sbp, "DBP": dbp, "PP": pp, "MAP": map_}


def stroke_volume(Vlv: np.ndarray, t: np.ndarray) -> Dict[str, float]:
    """
    Stroke volume and related PV quantities over the provided segment.

    Stroke volume is approximated as:
      SV = max(Vlv) - min(Vlv)

    Returns:
      - SV
      - Vmax
      - Vmin
    """
    Vlv = np.asarray(Vlv, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)
    if Vlv.size != t.size:
        raise ValueError("Vlv and t must have same length.")

    vmax = float(np.max(Vlv))
    vmin = float(np.min(Vlv))
    sv = vmax - vmin
    return {"SV": sv, "Vmax": vmax, "Vmin": vmin}


def ventricular_pressure_metrics(pLV: np.ndarray, t: np.ndarray) -> Dict[str, float]:
    """
    Basic ventricular pressure metrics.
    Returns:
      - pLV_max
      - pLV_min
      - pLV_mean (time average)
    """
    pLV = np.asarray(pLV, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)
    if pLV.size != t.size:
        raise ValueError("pLV and t must have same length.")

    return {
        "pLV_max": float(np.max(pLV)),
        "pLV_min": float(np.min(pLV)),
        "pLV_mean": _trapz_mean(pLV, t),
    }


def flow_metrics(Q2: np.ndarray, t: np.ndarray) -> Dict[str, float]:
    """
    Basic flow metrics on Q2.
    Returns:
      - Q2_mean (time average)
      - Q2_max
      - Q2_min
      - Q2_amp (half peak-to-peak)
    """
    Q2 = np.asarray(Q2, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)
    if Q2.size != t.size:
        raise ValueError("Q2 and t must have same length.")

    qmax = float(np.max(Q2))
    qmin = float(np.min(Q2))
    qmean = _trapz_mean(Q2, t)
    qamp = 0.5 * (qmax - qmin)
    return {"Q2_mean": qmean, "Q2_max": qmax, "Q2_min": qmin, "Q2_amp": qamp}


def valve_timing_metrics(
    flow: np.ndarray,
    t: np.ndarray,
    rel_threshold: float = 0.01,
) -> Dict[str, float]:
    """
    Compute simple valve timing metrics from a flow waveform.

    We define "open" when flow > rel_threshold * max(flow).
    This is a pragmatic definition that works well with smoothed valves.

    Returns:
      - peak: max flow
      - t_peak: time at max flow
      - open_duration: total time where valve is considered open
      - open_fraction: open_duration / cycle_duration
    """
    flow = np.asarray(flow, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)
    if flow.size != t.size:
        raise ValueError("flow and t must have same length.")
    if rel_threshold <= 0:
        raise ValueError("rel_threshold must be > 0.")

    peak = float(np.max(flow))
    if peak <= 0:
        # Valve effectively never opens
        return {"peak": peak, "t_peak": float(t[0]), "open_duration": 0.0, "open_fraction": 0.0}

    thr = rel_threshold * peak
    is_open = flow > thr

    # total open duration via trapezoidal integration on boolean mask
    # (approximate: sum of dt where open)
    dt = np.diff(t)
    # For each interval [i,i+1], consider open if both endpoints open
    open_intervals = is_open[:-1] & is_open[1:]
    open_duration = float(np.sum(dt[open_intervals]))
    cycle_duration = float(t[-1] - t[0])
    open_fraction = open_duration / cycle_duration if cycle_duration > 0 else 0.0

    idx_peak = int(np.argmax(flow))
    t_peak = float(t[idx_peak])

    return {"peak": peak, "t_peak": t_peak, "open_duration": open_duration, "open_fraction": open_fraction}


def compute_all_metrics(
    signals: Dict[str, np.ndarray],
    t: np.ndarray,
    Tcc: float,
    valve_threshold: float = 0.01,
) -> Dict[str, float]:
    """
    Compute a consolidated set of metrics on the last cardiac cycle.

    Expects signals to contain at least:
      - p1, pLV, Vlv, Q2, P0, P1

    Returns a flat dict of floats, suitable to store in SimulationResult.metrics.
    """
    required = ("p1", "pLV", "Vlv", "Q2", "P0", "P1")
    missing = [k for k in required if k not in signals]
    if missing:
        raise KeyError(f"Missing required signals: {missing}")

    t = np.asarray(t, dtype=float).reshape(-1)
    mask = _last_cycle_mask(t, Tcc)

    t_seg = t[mask]
    p1_seg = signals["p1"][mask]
    pLV_seg = signals["pLV"][mask]
    Vlv_seg = signals["Vlv"][mask]
    Q2_seg = signals["Q2"][mask]
    P0_seg = signals["P0"][mask]
    P1_seg = signals["P1"][mask]

    out: Dict[str, float] = {}

    out.update({f"p1_{k}": v for k, v in arterial_pressure_metrics(p1_seg, t_seg).items()})
    out.update(ventricular_pressure_metrics(pLV_seg, t_seg))
    out.update(stroke_volume(Vlv_seg, t_seg))
    out.update(flow_metrics(Q2_seg, t_seg))

    mv = valve_timing_metrics(P0_seg, t_seg, rel_threshold=valve_threshold)
    av = valve_timing_metrics(P1_seg, t_seg, rel_threshold=valve_threshold)

    out.update({f"MV_{k}": v for k, v in mv.items()})
    out.update({f"AV_{k}": v for k, v in av.items()})

    return out
