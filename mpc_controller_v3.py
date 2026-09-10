"""Conservative dual-input MPC for Prusa overhang commissioning.

This controller varies two inputs only:
    u = [print_speed_reference_mm_s, nozzle_setpoint_C]

Fan remains 100%, dwell remains zero, and the unvalidated dose objective is not
used. Both inputs are kept near the 100 mm/s, 210 C conventional reference with
narrow absolute and layer-to-layer limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from thermal_model import ThermalModelParams


@dataclass
class MPCConfig:
    """Settings for conservative dual-input commissioning MPC."""

    horizon: int = 2
    segment_length_mm: float = 300.0
    n_substeps: int = 12

    # Narrow actuator ranges around the conventional reference.
    v_min: float = 90.0
    v_max: float = 110.0
    T_nozzle_min: float = 205.0
    T_nozzle_max: float = 215.0

    # Hard and soft layer-to-layer movement limits.
    dv_max: float = 5.0
    dT_nozzle_max: float = 1.5

    # Thermal reference and deadband from the conventional 70-degree run.
    T_target_c: float = 46.6
    T_deadband_c: float = 0.75

    # Preferred conventional operating point.
    v_nominal: float = 100.0
    T_nozzle_nominal: float = 210.0

    # Objective weights.
    w_T: float = 1.0
    w_du: float = 0.50
    w_u_nominal: float = 0.25
    w_rate_violation: float = 20.0

    maxiter: int = 100
    ftol: float = 1e-6


@dataclass
class ControllerState:
    """Current filtered ROI temperature and previously applied input."""

    T: float
    u_prev: np.ndarray


def _deadband_error(value: float, target: float, deadband: float) -> float:
    raw = float(value) - float(target)
    return float(np.sign(raw) * max(abs(raw) - float(deadband), 0.0))


def empirical_next_temperature(
    current_temperature_c: float,
    control: Sequence[float],
) -> float:
    """
    Predict the next-layer ROI temperature using the two
    75-degree identification experiments.

    control = [speed_mm_s, nozzle_temperature_c]
    """
    speed_mms = float(control[0])
    nozzle_temperature_c = float(control[1])

    next_temperature_c = (
        -34.23
        + 0.754 * float(current_temperature_c)
        + 0.0513 * speed_mms
        + 0.1889 * nozzle_temperature_c
    )

    return float(next_temperature_c)


def _normalized_rate_violation(
    u: np.ndarray,
    u_prev: np.ndarray,
    cfg: MPCConfig,
) -> float:
    limits = np.array([cfg.dv_max, cfg.dT_nozzle_max], dtype=float)
    limits = np.maximum(limits, 1e-9)
    excess = np.maximum(np.abs(u - u_prev) - limits, 0.0) / limits
    return float(np.dot(excess, excess))


def _rollout_cost(
    z: np.ndarray,
    state: ControllerState,
    alpha_deg: float,
    model: ThermalModelParams,
    cfg: MPCConfig,
) -> float:
    """Roll the thermal model forward and evaluate tracking/control cost."""

    U = np.asarray(z, dtype=float).reshape(cfg.horizon, 2)
    T = float(state.T)
    u_last = np.asarray(state.u_prev, dtype=float).copy()

    movement_scale = np.array(
        [max(cfg.dv_max, 1e-9), max(cfg.dT_nozzle_max, 1e-9)],
        dtype=float,
    )
    nominal_scale = np.array(
        [
            max(0.5 * (cfg.v_max - cfg.v_min), 1e-9),
            max(0.5 * (cfg.T_nozzle_max - cfg.T_nozzle_min), 1e-9),
        ],
        dtype=float,
    )
    u_nominal = np.array([cfg.v_nominal, cfg.T_nozzle_nominal], dtype=float)

    cost = 0.0

    for u_i in U:
        T = empirical_next_temperature(
            current_temperature_c=T,
            control=u_i,
        )

        temperature_error = _deadband_error(
            T,
            cfg.T_target_c,
            cfg.T_deadband_c,
        )
        du_normalized = (u_i - u_last) / movement_scale
        nominal_error = (u_i - u_nominal) / nominal_scale

        cost += cfg.w_T * temperature_error**2
        cost += cfg.w_du * float(np.dot(du_normalized, du_normalized))
        cost += cfg.w_u_nominal * float(np.dot(nominal_error, nominal_error))
        cost += cfg.w_rate_violation * _normalized_rate_violation(
            u_i,
            u_last,
            cfg,
        )

        u_last = u_i.copy()

    return float(cost)


def _hard_limit_action(
    proposed: np.ndarray,
    previous: np.ndarray,
    measured_T: float,
    cfg: MPCConfig,
) -> np.ndarray:
    """Apply absolute/rate limits and a conservative thermal direction guard."""

    proposed = np.asarray(proposed, dtype=float).copy()
    previous = np.asarray(previous, dtype=float).copy()

    # Absolute limits first.
    proposed[0] = float(np.clip(proposed[0], cfg.v_min, cfg.v_max))
    proposed[1] = float(
        np.clip(proposed[1], cfg.T_nozzle_min, cfg.T_nozzle_max)
    )

    # Hard layer-to-layer limits.
    proposed[0] = float(
        np.clip(
            proposed[0],
            max(cfg.v_min, previous[0] - cfg.dv_max),
            min(cfg.v_max, previous[0] + cfg.dv_max),
        )
    )
    proposed[1] = float(
        np.clip(
            proposed[1],
            max(cfg.T_nozzle_min, previous[1] - cfg.dT_nozzle_max),
            min(cfg.T_nozzle_max, previous[1] + cfg.dT_nozzle_max),
        )
    )

    # ------------------------------------------------------------
    # Direction guard for the fitted 75-degree empirical model
    # ------------------------------------------------------------
    #
    # In the fitted model, both a higher speed reference and a
    # higher nozzle temperature increase the predicted next-layer
    # ROI temperature.
    #
    # Below target:
    #   do not decrease speed
    #   do not decrease nozzle temperature
    #
    # Above target:
    #   do not increase speed
    #   do not increase nozzle temperature
    #
    # Inside the deadband:
    #   hold both commands to prevent chatter.
    if measured_T < cfg.T_target_c - cfg.T_deadband_c:
        proposed[0] = max(
            proposed[0],
            previous[0],
        )
        proposed[1] = max(
            proposed[1],
            previous[1],
        )

    elif measured_T > cfg.T_target_c + cfg.T_deadband_c:
        proposed[0] = min(
            proposed[0],
            previous[0],
        )
        proposed[1] = min(
            proposed[1],
            previous[1],
        )

    else:
        proposed = previous.copy()

    return proposed


def solve_mpc(
    state: ControllerState,
    alpha_deg: float,
    model: ThermalModelParams,
    cfg: MPCConfig,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Return the first bounded dual-input action and solver diagnostics."""

    previous = np.asarray(state.u_prev, dtype=float).copy()
    previous[0] = float(np.clip(previous[0], cfg.v_min, cfg.v_max))
    previous[1] = float(
        np.clip(previous[1], cfg.T_nozzle_min, cfg.T_nozzle_max)
    )

    z0 = np.tile(previous, cfg.horizon)
    bounds = []
    for _ in range(cfg.horizon):
        bounds.extend(
            [
                (cfg.v_min, cfg.v_max),
                (cfg.T_nozzle_min, cfg.T_nozzle_max),
            ]
        )

    result = minimize(
        _rollout_cost,
        z0,
        args=(state, float(alpha_deg), model, cfg),
        method="SLSQP",
        bounds=bounds,
        options={
            "maxiter": cfg.maxiter,
            "ftol": cfg.ftol,
            "disp": False,
        },
    )

    if result.success and np.isfinite(result.fun):
        proposed = np.asarray(result.x, dtype=float).reshape(cfg.horizon, 2)[0]
        fallback = False
    else:
        proposed = previous.copy()
        fallback = True

    applied = _hard_limit_action(
        proposed,
        previous,
        float(state.T),
        cfg,
    )

    predicted_next_T = empirical_next_temperature(
        current_temperature_c=float(state.T),
        control=applied,
    )

    # The empirical transition model predicts one layer ahead
    # but does not predict physical layer duration.
    predicted_dt = float("nan")

    info: Dict[str, object] = {
        "success": bool(result.success) and not fallback,
        "message": str(result.message),
        "cost": float(result.fun) if np.isfinite(result.fun) else np.nan,
        "n_iter": int(result.nit) if hasattr(result, "nit") else None,
        "fallback": fallback,
        "proposed_speed_mms": float(proposed[0]),
        "proposed_nozzle_c": float(proposed[1]),
        "applied_speed_mms": float(applied[0]),
        "applied_nozzle_c": float(applied[1]),
        "predicted_next_T_c": float(predicted_next_T),
        "predicted_dt_s": float(predicted_dt),
    }

    return applied, info
