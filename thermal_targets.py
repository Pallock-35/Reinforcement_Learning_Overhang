"""Thermal target and dose-window functions (outline §3.3)."""

from __future__ import annotations

from typing import Tuple

# Reference overhang angle for linear schedules (deg).
ALPHA_REF = 45.0

# Temperature target: T*(α) = T_ref - slope_T * max(α - α_ref, 0)
T_REF = 145.0
T_SLOPE = 0.45

# Soft temperature constraints (outline §6.5).
T_BOND_REF = 95.0
T_SAG_REF = 175.0
T_CONSTRAINT_SLOPE = 0.35

# Dose target uses absolute window width ΔD (outline §3.3), not ±25%.
# D_0 is scaled to a nominal layer interval:
#   dt_nom ≈ layer_length_mm / v_nom = 300 / 35 ≈ 8.57 s
#   D_0 ≈ dt_nom * (T_REF - T_g) ≈ 8.57 * 85 ≈ 728
D_REF = 730.0
D_SLOPE = 6.0
DELTA_D = 180.0


def temperature_target(alpha_deg: float) -> float:
    """Target IR temperature as a function of overhang angle (deg C)."""
    return T_REF - T_SLOPE * max(alpha_deg - ALPHA_REF, 0.0)


def temperature_bond_limit(alpha_deg: float) -> float:
    """Lower IR temperature bound for bonding risk (deg C)."""
    return T_BOND_REF - T_CONSTRAINT_SLOPE * max(alpha_deg - ALPHA_REF, 0.0)


def temperature_sag_limit(alpha_deg: float) -> float:
    """Upper IR temperature bound for sag risk (deg C)."""
    return T_SAG_REF - T_CONSTRAINT_SLOPE * max(alpha_deg - ALPHA_REF, 0.0)


def dose_target(alpha_deg: float) -> float:
    """
    Target per-decision thermal dose above T_g (deg C·s).

    D*(α) = D_0 - c_D (α - α_0), matching outline §3.3.
    Dose is the integral over one layer/decision interval, not the full print.
    """
    return D_REF - D_SLOPE * max(alpha_deg - ALPHA_REF, 0.0)


def dose_window(alpha_deg: float) -> Tuple[float, float]:
    """Thermal-dose lower and upper bounds: D* ± ΔD (outline §3.3)."""
    d_star = dose_target(alpha_deg)
    return max(d_star - DELTA_D, 0.0), d_star + DELTA_D
