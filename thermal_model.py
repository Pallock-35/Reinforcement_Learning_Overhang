"""Reduced-order analytical thermal model (outline §4–5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares


class ThermalStepConfig(Protocol):
    """Duck-typed config with actuator bounds and segment length."""

    segment_length_mm: float
    v_min: float
    v_max: float
    T_nozzle_min: float
    T_nozzle_max: float


@dataclass
class ThermalModelParams:
    """Parameters of the reduced-order analytical thermal model."""

    T_inf: float = 25.0
    T_nozzle: float = 215.0
    T_g: float = 60.0

    beta: float = 0.35
    a_v: float = 0.15
    tau0: float = 8.0
    a_fan: float = 2.0
    a_alpha: float = 0.4


def analytical_thermal_step(
    T: float,
    D: float,
    u: Sequence[float],
    alpha_deg: float,
    model: ThermalModelParams,
    cfg: ThermalStepConfig,
) -> Tuple[float, float, float]:
    """One-step analytical thermal model; returns (T_next, D_next, dt).

    Control vector ``u = [v_mms, T_nozzle_C]``. Fan cooling is locked at 100%;
    dwell is always 0.
    """
    v, T_nozzle = map(float, u)

    v = float(np.clip(v, cfg.v_min, cfg.v_max))
    T_nozzle = float(np.clip(T_nozzle, cfg.T_nozzle_min, cfg.T_nozzle_max))
    fan = 1.0

    dt = cfg.segment_length_mm / max(v, 1e-9)

    alpha_norm = max(alpha_deg, 0.0) / 90.0

    tau_eff = model.tau0 / (1.0 + model.a_fan * fan + model.a_alpha * alpha_norm)
    tau_eff = max(tau_eff, 1e-6)

    T_deposit_eff = T_nozzle - model.a_v * v
    T_plus = (1.0 - model.beta) * T + model.beta * T_deposit_eff

    T_next = model.T_inf + (T_plus - model.T_inf) * np.exp(-dt / tau_eff)

    T_avg = 0.5 * (T + T_next)
    D_next = D + dt * max(T_avg - model.T_g, 0.0)

    return float(T_next), float(D_next), float(dt)


@dataclass
class _SubstepConfig:
    """Lightweight config for sub-segment integration."""

    segment_length_mm: float
    v_min: float
    v_max: float
    T_nozzle_min: float
    T_nozzle_max: float


def analytical_layer_step(
    T: float,
    u: Sequence[float],
    alpha_deg: float,
    model: ThermalModelParams,
    cfg: ThermalStepConfig,
    *,
    n_substeps: int = 12,
) -> Tuple[float, float, float]:
    """
    Integrate the segment model over one layer-scale decision interval.

    Continuous deposition is approximated by N short analytical steps so
    temperature does not collapse to ambient over a long layer dt.
    Returns (T_next, D_layer, dt_total).
    """
    n_substeps = max(int(n_substeps), 1)
    layer_length = float(cfg.segment_length_mm)
    sub_length = layer_length / n_substeps

    sub_cfg = _SubstepConfig(
        segment_length_mm=sub_length,
        v_min=cfg.v_min,
        v_max=cfg.v_max,
        T_nozzle_min=cfg.T_nozzle_min,
        T_nozzle_max=cfg.T_nozzle_max,
    )

    v, T_nozzle = map(float, u)
    T_cur = float(T)
    D_layer = 0.0
    dt_total = 0.0

    for _ in range(n_substeps):
        T_cur, D_layer, dt_i = analytical_thermal_step(
            T_cur,
            D_layer,
            [v, T_nozzle],
            alpha_deg,
            model,
            sub_cfg,
        )
        dt_total += dt_i

    return float(T_cur), float(D_layer), float(dt_total)


def pack_params(model: ThermalModelParams) -> np.ndarray:
    """Convert model parameters to vector for least-squares fitting."""
    return np.array(
        [model.beta, model.a_v, model.tau0, model.a_fan, model.a_alpha],
        dtype=float,
    )


def unpack_params(theta: Sequence[float], base: ThermalModelParams) -> ThermalModelParams:
    """Create model parameters from vector while preserving fixed values."""
    beta, a_v, tau0, a_fan, a_alpha = map(float, theta)
    return ThermalModelParams(
        T_inf=base.T_inf,
        T_nozzle=base.T_nozzle,
        T_g=base.T_g,
        beta=beta,
        a_v=a_v,
        tau0=tau0,
        a_fan=a_fan,
        a_alpha=a_alpha,
    )


def fit_thermal_model(
    calibration_rows: List[Dict[str, float]],
    base_model: ThermalModelParams,
    cfg: ThermalStepConfig,
    *,
    n_substeps: int = 12,
) -> ThermalModelParams:
    """Fit thermal model parameters from calibration data."""
    theta0 = pack_params(base_model)

    lower = np.array([0.01, 0.0, 0.5, 0.0, -2.0])
    upper = np.array([0.99, 2.0, 60.0, 10.0, 5.0])

    def residuals(theta: np.ndarray) -> np.ndarray:
        trial_model = unpack_params(theta, base_model)
        errors = []
        for row in calibration_rows:
            T_nozzle = float(row.get("T_nozzle", row.get("nozzle", base_model.T_nozzle)))
            T_pred, _, _ = analytical_layer_step(
                T=row["T_k"],
                u=[row["v"], T_nozzle],
                alpha_deg=row["alpha_deg"],
                model=trial_model,
                cfg=cfg,
                n_substeps=n_substeps,
            )
            errors.append(T_pred - row["T_next_measured"])
        return np.array(errors, dtype=float)

    result = least_squares(
        residuals,
        theta0,
        bounds=(lower, upper),
        max_nfev=500,
    )

    return unpack_params(result.x, base_model)


def model_params_to_dict(model: ThermalModelParams) -> dict[str, float]:
    """Serialize model parameters to a JSON-friendly dict."""
    return {
        "T_inf": model.T_inf,
        "T_nozzle": model.T_nozzle,
        "T_g": model.T_g,
        "beta": model.beta,
        "a_v": model.a_v,
        "tau0": model.tau0,
        "a_fan": model.a_fan,
        "a_alpha": model.a_alpha,
    }


def model_params_from_dict(payload: dict[str, float]) -> ThermalModelParams:
    """Load model parameters from a dict."""
    return ThermalModelParams(
        T_inf=float(payload.get("T_inf", 25.0)),
        T_nozzle=float(payload.get("T_nozzle", 215.0)),
        T_g=float(payload.get("T_g", 60.0)),
        beta=float(payload.get("beta", 0.35)),
        a_v=float(payload.get("a_v", 0.15)),
        tau0=float(payload.get("tau0", 8.0)),
        a_fan=float(payload.get("a_fan", 2.0)),
        a_alpha=float(payload.get("a_alpha", 0.4)),
    )
