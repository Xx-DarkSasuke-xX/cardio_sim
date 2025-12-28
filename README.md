# Cardiovascular Systemic Circulation Simulator (0D)

A lumped-parameter model of the systemic circulation coupling left ventricular dynamics with an arterial Windkessel network.

---

## Introduction

This project implements a zero-dimensional (0D) cardiovascular model in Python. The model describes the interaction between the left ventricle and the systemic arterial tree using ordinary differential equations derived from hydraulic analogies. It was developed as part of coursework at the École Polytechnique de Bruxelles.

The simulator captures key hemodynamic phenomena:
- Time-varying ventricular elastance (contraction/relaxation cycle)
- Nonlinear valve dynamics (mitral and aortic)
- Arterial compliance and peripheral resistance (Windkessel approach)

The codebase supports both nonlinear time-domain simulation and linearized system-theoretic analysis (observability, identifiability).

---

## Mathematical Model

### State variables

The system evolves three state variables:

| Symbol | Description | Unit |
|--------|-------------|------|
| $p_{LV}$ | Left ventricular pressure | mmHg |
| $p_1$ | Aortic (arterial) pressure | mmHg |
| $Q_2$ | Peripheral arterial flow | mL/s |

### Governing equations

The dynamics follow from mass and momentum conservation in a lumped framework:

\begin{equation}
\begin{cases}
C_{LV}(t)\,\dfrac{dp_{LV}}{dt}
+ p_{LV}(t)\,\dfrac{dC_{LV}}{dt}
= \alpha(t) - \beta(t), \\[6pt]
I_{\mathrm{art}}\,\dfrac{dQ_2}{dt}
= p_1(t) - p_{RA}
- \left(R_{h,\mathrm{cap}} + R_{h,\mathrm{art}}\right) Q_2(t), \\[6pt]
C_{\mathrm{art}}\,\dfrac{dp_1}{dt}
= \beta(t) - Q_2(t).
\end{cases}
\label{eq:global_ode_system}
\end{equation}

where $C_{LV}(t)$ is a prescribed time-varying compliance representing the cardiac cycle, and valve flows $\alpha(t)$ and $\beta(t)$ are modelled using smoothed Heaviside functions to avoid discontinuities.

### Linearized arterial subsystem

For system-theoretic analysis, the arterial network (downstream of the aortic valve) is linearized around a steady operating point. This yields a second-order transfer function relating inlet flow perturbations to arterial pressure, enabling classical observability and identifiability studies.

---

## Installation

**Requirements:** Python ≥ 3.8

```bash
pip install numpy scipy matplotlib
```

Clone the repository and run scripts directly—no package installation is required.

---

## Usage

### Simulate a healthy cardiac cycle

```bash
python scripts/run_healthy.py
```

This runs 10 cardiac cycles, prints hemodynamic metrics (SBP, DBP, MAP, stroke volume), and displays pressure–volume loops and waveform plots.

### Compare healthy vs. pathological conditions

```bash
python scripts/run_compare_pathology.py
```

Pathology is modelled by altering arterial compliance and resistance (e.g., arterial stiffening). Both scenarios are simulated and compared side by side.

### Linear system analysis

```bash
python scripts/run_lti_analysis.py
```

Computes poles, zeros, Bode plots, observability matrix rank, and identifiability metrics for the linearized arterial model.

### Export figures

```bash
python scripts/export_figures.py
python scripts/export_lti_figures.py
```

Figures are saved to `exports/figures/` and `exports/figures_lti/`.

---

## Project layout

```
cardio/
├── config/       # Simulation settings, default parameters
├── params/       # Parameter sets (healthy, pathology)
├── physiology/   # Activation function, compliance, valve models
├── models/       # ODE right-hand side, signal reconstruction
├── simulation/   # Integrators, initial conditions, pipeline
├── analysis/     # Metrics, linearization, observability, identifiability
└── plotting/     # Waveform and LTI visualisation

scripts/          # Entry points for common tasks
tests/            # Unit tests (pytest)
exports/          # Generated figures
```

---

## Testing

```bash
pytest tests/
```

Tests cover activation functions, compliance models, and ODE right-hand side behaviour at physiological limits.

---

## Authors

| Name | Email | Affiliation |
|------|-------|-------------|
| Tahar Mansouri | tahar.mansouri@ulb.be | École Polytechnique de Bruxelles, ULB |
| Joshua Otu | joshua.otu@ulb.be | École Polytechnique de Bruxelles, ULB |
| Mamadou Tambassa | mamadou.tambassa@ulb.be | École Polytechnique de Bruxelles, ULB |

---

## License

This project was developed for educational purposes. Contact the authors for reuse or collaboration.


