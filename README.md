
# Cardiovascular Systemic Circulation Simulator (0D)

## Overview
This repository contains a modular Python implementation of a lumped-parameter (0D) model for the systemic cardiovascular circulation. The model focuses on the interaction between the left ventricle and the arterial system, allowing simulation and comparison of healthy and pathological hemodynamics. The codebase is organized for clarity, reproducibility, and ease of extension.

## Features
- Simulates systemic circulation using a nonlinear ODE model
- Models left ventricular activation, time-varying compliance, and valve dynamics
- Supports healthy and pathological scenarios (e.g., arterial stiffening, increased afterload)
- Linearization tools for system-theoretic analysis (observability, identifiability)
- Generates time-resolved pressure, flow, and volume waveforms
- Produces publication-ready plots and exports figures
- Includes unit tests for physiological and numerical components

## Project Structure

- `cardio/config`: Default simulation and parameter settings
- `cardio/params`: Parameter sets for healthy and pathological cases
- `cardio/physiology`: Ventricular activation, compliance, and valve models
- `cardio/models`: System equations and signal reconstruction
- `cardio/simulation`: Integration routines and simulation pipelines
- `cardio/analysis`: Metrics, linearization, observability, identifiability
- `cardio/plotting`: Plotting utilities for waveforms and LTI analysis
- `scripts`: Example scripts for running simulations and exporting figures
- `tests`: Unit tests for model components
- `exports`: Output directory for generated figures

## How to Run Simulations

1. **Healthy scenario:**
	- Run `python scripts/run_healthy.py` to simulate and plot a healthy cardiac cycle.

2. **Pathology comparison:**
	- Run `python scripts/run_compare_pathology.py` to compare healthy and pathological (e.g., arterial stiffening) conditions. Key metrics and plots are shown for both cases.

3. **Linear analysis:**
	- Run `python scripts/run_lti_analysis.py` for linearized arterial system analysis (poles, zeros, observability, identifiability, LTI plots).

4. **Exporting figures:**
	- Use `python scripts/export_figures.py` and `python scripts/export_lti_figures.py` to generate and save figures to the `exports/` directory.

## Model Details

- **State variables:**
  - Left ventricular pressure (pLV)
  - Peripheral arterial flow (Q2)
  - Aortic pressure (p1)
- **Auxiliary signals:**
  - Left ventricular volume (Vlv)
  - Ventricular compliance (Clv) and elastance (Elv)
  - Mitral and aortic valve flows (P0, P1)
- **Measured output:**
  - Arterial pressure (p1) is the main observable

## Pathology Modeling
Pathological scenarios are created by modifying arterial compliance and resistance parameters, representing conditions such as arteriosclerosis. The model structure remains unchanged, ensuring that differences reflect only parameter changes. Pathology parameters are configurable in `cardio/params/pathology.py`.

## Numerical Approach
- Integrates the nonlinear ODE system over multiple cardiac cycles
- Smooths valve transitions for numerical stability
- Analyzes the final converged cycle for metrics and plots

## Testing
Unit tests are provided in the `tests/` directory. Run with your preferred test runner, e.g.:

```
pytest tests/
```

## Requirements
- Python 3.8+
- numpy, scipy, matplotlib

## Roadmap
- Interfaces and dataclasses for parameters and results
- Global simulation pipeline
- Physiological equations and integration
- Pathology scenarios and system-theoretic analysis


