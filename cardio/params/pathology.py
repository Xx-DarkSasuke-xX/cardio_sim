
from __future__ import annotations

from dataclasses import replace

from cardio.params.dataclasses import ParameterSet


def reduced_arterial_compliance(
    base: ParameterSet,
    factor: float = 0.3,  # <-- CHANGEMENT: défaut plus "stiffness visible" (au lieu de 0.5)
    label: str = "pathology_reduced_compliance",
    *,
    inertance_factor: float = 1.0,  # <-- AJOUT: optionnel, par défaut ne change rien
) -> ParameterSet:
    """
    Pathology transform: reduce arterial compliance (arterial stiffness PURE if used alone).

    Typical use:
      - factor = 0.5 -> mild stiffening
      - factor = 0.3 -> moderate stiffening (recommended default to see an effect)
      - factor = 0.2 -> severe stiffening

    Optional (dynamic accent):
      - inertance_factor in [1.0, 1.5] to sharpen the pressure upstroke.
      - Keep modest to avoid oscillations depending on the solver.

    Model note:
      - Total peripheral resistance is Rtot = Rart + Rcap (unchanged here),
        so this transform targets pulsatility (PP) rather than pure mean pressure.
    """
    if factor <= 0.0:
        raise ValueError("factor must be > 0")
    if inertance_factor <= 0.0:
        raise ValueError("inertance_factor must be > 0")

    return replace(
        base,
        Cart=base.Cart * factor,
        Iart=base.Iart * inertance_factor,  # <-- AJOUT: n'affecte rien si inertance_factor=1.0
        label=label,
        meta={
            **dict(base.meta),
            "pathology": "reduced_arterial_compliance",
            "factor": factor,
            "inertance_factor": inertance_factor,
        },
    )


def increased_afterload(
    base: ParameterSet,
    factor: float = 1.5,
    label: str = "pathology_increased_afterload",
    *,
    scale_capillary: bool = True,
) -> ParameterSet:
    """
    Pathology transform: increase peripheral resistance (afterload).

    Project definition:
      - Total peripheral resistance is Rtot = Rart + Rcap
    """
    if factor <= 0.0:
        raise ValueError("factor must be > 0")

    updates = {"Rart": base.Rart * factor}
    if scale_capillary:
        updates["Rcap"] = base.Rcap * factor

    return replace(
        base,
        **updates,
        label=label,
        meta={
            **dict(base.meta),
            "pathology": "increased_afterload",
            "factor": factor,
            "scale_capillary": scale_capillary,
        },
    )


def combined_stiffness_and_afterload(
    base: ParameterSet,
    compliance_factor: float = 0.5,
    resistance_factor: float = 1.5,
    label: str = "pathology_combined",
    *,
    scale_capillary: bool = True,
) -> ParameterSet:
    """
    Combine Cart↓ and Rtot↑ (NOT pure stiffness).
    """
    tmp = reduced_arterial_compliance(base, factor=compliance_factor, label=label)
    return increased_afterload(
        tmp,
        factor=resistance_factor,
        label=label,
        scale_capillary=scale_capillary,
    )


def arterial_stiffening_combo(
    base: ParameterSet,
    compliance_factor: float = 0.3,
    resistance_factor: float = 2.5,
    inertance_factor: float = 1.5,
    label: str = "pathology_stiffening_combo",
    *,
    scale_capillary: bool = True,
) -> ParameterSet:
    """
    Stiffening + afterload + inertance (NOT pure stiffness).
    """
    if compliance_factor <= 0.0:
        raise ValueError("compliance_factor must be > 0")
    if resistance_factor <= 0.0:
        raise ValueError("resistance_factor must be > 0")
    if inertance_factor <= 0.0:
        raise ValueError("inertance_factor must be > 0")

    updates = {
        "Cart": base.Cart * compliance_factor,
        "Rart": base.Rart * resistance_factor,
        "Iart": base.Iart * inertance_factor,
    }
    if scale_capillary:
        updates["Rcap"] = base.Rcap * resistance_factor

    return replace(
        base,
        **updates,
        label=label,
        meta={
            **dict(base.meta),
            "pathology": "stiffening_combo",
            "compliance_factor": compliance_factor,
            "resistance_factor": resistance_factor,
            "inertance_factor": inertance_factor,
            "scale_capillary": scale_capillary,
        },
    )
