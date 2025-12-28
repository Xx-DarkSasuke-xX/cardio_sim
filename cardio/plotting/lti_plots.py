from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def plot_pole_zero_map(
    poles: np.ndarray,
    zeros: np.ndarray,
    title: str = "Pole-zero map",
    ax: Optional[plt.Axes] = None,
    show_grid: bool = True,
) -> plt.Axes:
    """
    Plot pole-zero map in the complex plane.

    poles, zeros: arrays of complex numbers
    """
    if ax is None:
        _, ax = plt.subplots()

    poles = np.asarray(poles, dtype=complex)
    zeros = np.asarray(zeros, dtype=complex)

    if zeros.size > 0:
        ax.scatter(np.real(zeros), np.imag(zeros), marker="o", facecolors="none", edgecolors="k", label="zeros")
    if poles.size > 0:
        ax.scatter(np.real(poles), np.imag(poles), marker="x", color="k", label="poles")

    ax.axhline(0.0, linewidth=1)
    ax.axvline(0.0, linewidth=1)

    ax.set_xlabel("Real")
    ax.set_ylabel("Imag")
    ax.set_title(title)
    if show_grid:
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(loc="best")
    return ax


def plot_impulse_response(
    sys_tf: signal.TransferFunction,
    t_end: float = 5.0,
    n: int = 2000,
    title: str = "Impulse response (Δp1 from ΔQin)",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Impulse response y(t) for the transfer function sys_tf.
    """
    if ax is None:
        _, ax = plt.subplots()

    t = np.linspace(0.0, float(t_end), int(n))
    tout, y = signal.impulse(sys_tf, T=t)

    ax.plot(tout, y)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Δp1 (arb. units)")
    ax.grid(True, linestyle="--", linewidth=0.5)
    return ax


def plot_step_response(
    sys_tf: signal.TransferFunction,
    t_end: float = 5.0,
    n: int = 2000,
    title: str = "Step response (Δp1 from ΔQin)",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Step response y(t) for the transfer function sys_tf.
    """
    if ax is None:
        _, ax = plt.subplots()

    t = np.linspace(0.0, float(t_end), int(n))
    tout, y = signal.step(sys_tf, T=t)

    ax.plot(tout, y)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Δp1 (arb. units)")
    ax.grid(True, linestyle="--", linewidth=0.5)
    return ax


def plot_bode(
    sys_tf: signal.TransferFunction,
    w: Optional[np.ndarray] = None,
    w_min: float = 1e-2,
    w_max: float = 1e3,
    n: int = 1000,
    title: str = "Bode plot",
    ax_mag: Optional[plt.Axes] = None,
    ax_phase: Optional[plt.Axes] = None,
    magnitude_db: bool = True,
) -> Tuple[plt.Axes, plt.Axes]:
    """
    Bode magnitude + phase for sys_tf using scipy.signal.freqresp.

    w: rad/s frequency grid. If None, logspace(w_min, w_max, n).
    """
    if w is None:
        w = np.logspace(np.log10(float(w_min)), np.log10(float(w_max)), int(n))

    w = np.asarray(w, dtype=float)
    _, H = signal.freqresp(sys_tf, w=w)
    mag = np.abs(H)
    phase = np.angle(H, deg=True)

    if ax_mag is None or ax_phase is None:
        fig, (ax_mag, ax_phase) = plt.subplots(2, 1, sharex=True)
        fig.suptitle(title)

    if magnitude_db:
        mag_plot = 20.0 * np.log10(np.maximum(mag, 1e-30))
        ax_mag.semilogx(w, mag_plot)
        ax_mag.set_ylabel("Magnitude (dB)")
    else:
        ax_mag.semilogx(w, mag)
        ax_mag.set_ylabel("Magnitude")

    ax_phase.semilogx(w, phase)
    ax_phase.set_ylabel("Phase (deg)")
    ax_phase.set_xlabel("Frequency ω (rad/s)")

    ax_mag.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax_phase.grid(True, which="both", linestyle="--", linewidth=0.5)

    return ax_mag, ax_phase


def plot_lti_summary(
    poles: np.ndarray,
    zeros: np.ndarray,
    sys_tf: signal.TransferFunction,
    title_prefix: str = "Arterial Windkessel (LTI)",
    t_end: float = 5.0,
) -> None:
    """
    Convenience function to generate the standard set of LTI plots:
    - pole-zero map
    - impulse response
    - step response
    - bode plot
    """
    plot_pole_zero_map(poles, zeros, title=f"{title_prefix} — Pole-zero map")
    plot_impulse_response(sys_tf, t_end=t_end, title=f"{title_prefix} — Impulse response")
    plot_step_response(sys_tf, t_end=t_end, title=f"{title_prefix} — Step response")
    plot_bode(sys_tf, title=f"{title_prefix} — Bode plot")
