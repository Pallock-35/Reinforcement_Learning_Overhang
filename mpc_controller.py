"""MPC controller, objective, solver, and offline simulators (outline §6, §8)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from thermal_model import ThermalModelParams, analytical_layer_step
from thermal_targets import (
    dose_target,
    dose_window,
    temperature_bond_limit,
    temperature_sag_limit,
    temperature_target,
)


@dataclass
class MPCConfig:
    """MPC settings, actuator limits, and objective weights."""

    horizon: int = 6
    # Layer-scale path length so predicted dt matches live layer decisions.
    segment_length_mm: float = 300.0
    # Substeps approximate continuous deposition within one layer.
    n_substeps: int = 12

    # Input bounds aligned with M220 @ 170 mm/s base (50–170 mm/s)
    v_min: float = 50.0
    v_max: float = 170.0

    # PLA-safe nozzle setpoint bounds (°C)
    T_nozzle_min: float = 190.0
    T_nozzle_max: float = 230.0

    # Measured reference from the conventional print
    T_target_override_c: float | None = None

    dv_max: float = 20.0
    dT_nozzle_max: float = 15.0

    w_T: float = 1.0
    w_D: float = 0.02
    w_window: float = 100.0
    w_temp_constraint: float = 50.0
    w_du: float = 0.10
    w_rate_violation: float = 10.0

    maxiter: int = 100
    ftol: float = 1e-5


@dataclass
class ControllerState:
    """Current state used by the controller."""

    T: float
    D: float
    u_prev: np.ndarray


def conservative_fallback(cfg: MPCConfig) -> np.ndarray:
    """Cooling-biased action when the optimizer fails: slow + cooler nozzle."""
    return np.array(
        [
            cfg.v_min + 0.25 * (cfg.v_max - cfg.v_min),
            cfg.T_nozzle_min + 0.25 * (cfg.T_nozzle_max - cfg.T_nozzle_min),
        ],
        dtype=float,
    )


def soft_rate_violation(
    u: np.ndarray,
    u_prev: np.ndarray,
    cfg: MPCConfig,
) -> float:
    """Quadratic penalty for exceeding recommended actuator rate limits."""
    du = np.abs(u - u_prev)
    limits = np.array([cfg.dv_max, cfg.dT_nozzle_max], dtype=float)
    excess = np.maximum(du - limits, 0.0)
    return float(np.dot(excess, excess))


def mpc_objective(
    z: np.ndarray,
    state: ControllerState,
    future_alphas: Sequence[float],
    model: ThermalModelParams,
    cfg: MPCConfig,
) -> float:
    """
    Objective minimized by MPC over the prediction horizon.

    Temperature carries across the horizon. Thermal dose is evaluated
    per decision step (per-layer dose), matching live layerwise logging.
    Control vector per step: ``u = [v_mms, T_nozzle_C]``.
    """
    U = z.reshape(cfg.horizon, 2)

    T = float(state.T)
    u_last = np.array(state.u_prev, dtype=float)

    cost = 0.0

    for i in range(cfg.horizon):
        alpha_i = float(future_alphas[min(i, len(future_alphas) - 1)])
        u_i = U[i]

        # Per-layer dose via substepped deposition model.
        T, D_layer, _ = analytical_layer_step(
            T,
            u_i,
            alpha_i,
            model,
            cfg,
            n_substeps=cfg.n_substeps,
        )

        if cfg.T_target_override_c is not None:
            T_ref = float(cfg.T_target_override_c)
        else:
            T_ref = temperature_target(alpha_i)
        D_ref = dose_target(alpha_i)
        D_low, D_high = dose_window(alpha_i)
        T_bond = temperature_bond_limit(alpha_i)
        T_sag = temperature_sag_limit(alpha_i)

        T_error = T - T_ref
        D_error = D_layer - D_ref

        low_violation = max(D_low - D_layer, 0.0)
        high_violation = max(D_layer - D_high, 0.0)
        cold_violation = max(T_bond - T, 0.0)
        hot_violation = max(T - T_sag, 0.0)

        du = u_i - u_last
        rate_penalty = soft_rate_violation(u_i, u_last, cfg)

        cost += cfg.w_T * T_error**2
        cost += cfg.w_D * D_error**2
        cost += cfg.w_window * (low_violation**2 + high_violation**2)
        cost += cfg.w_temp_constraint * (cold_violation**2 + hot_violation**2)
        cost += cfg.w_du * float(np.dot(du, du))
        cost += cfg.w_rate_violation * rate_penalty

        u_last = u_i.copy()

    return float(cost)


def solve_mpc(
    state: ControllerState,
    future_alphas: Sequence[float],
    model: ThermalModelParams,
    cfg: MPCConfig,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Solve the MPC problem and return the first control action."""
    if len(future_alphas) == 0:
        raise ValueError("future_alphas must contain at least one value.")

    if len(future_alphas) < cfg.horizon:
        future_alphas = list(future_alphas) + [future_alphas[-1]] * (
            cfg.horizon - len(future_alphas)
        )

    bounds = []
    for _ in range(cfg.horizon):
        bounds.append((cfg.v_min, cfg.v_max))
        bounds.append((cfg.T_nozzle_min, cfg.T_nozzle_max))

    # Warm start from previous action (outline §6.7).
    z0 = np.tile(np.array(state.u_prev, dtype=float), cfg.horizon)
    z0 = np.clip(
        z0,
        np.array([cfg.v_min, cfg.T_nozzle_min] * cfg.horizon),
        np.array([cfg.v_max, cfg.T_nozzle_max] * cfg.horizon),
    )

    result = minimize(
        mpc_objective,
        z0,
        args=(state, future_alphas, model, cfg),
        method="SLSQP",
        bounds=bounds,
        options={
            "maxiter": cfg.maxiter,
            "ftol": cfg.ftol,
            "disp": False,
        },
    )

    if result.success and np.isfinite(result.fun):
        U_opt = result.x.reshape(cfg.horizon, 2)
        u_apply = U_opt[0].copy()
        fallback_used = False
    else:
        # Prefer previous action; if it is out of bounds use conservative cooling.
        u_prev = np.array(state.u_prev, dtype=float)
        in_bounds = (
            cfg.v_min <= u_prev[0] <= cfg.v_max
            and cfg.T_nozzle_min <= u_prev[1] <= cfg.T_nozzle_max
        )
        u_apply = u_prev if in_bounds else conservative_fallback(cfg)
        fallback_used = True

    info = {
        "success": bool(result.success) and not fallback_used,
        "message": str(result.message),
        "cost": float(result.fun) if np.isfinite(result.fun) else np.nan,
        "n_iter": int(result.nit) if hasattr(result, "nit") else None,
        "fallback": fallback_used,
    }

    return u_apply, info


def summarize_history(history: List[Dict[str, float]]) -> Dict[str, float]:
    """Compute outline §9 thermal performance metrics."""
    D = np.array([row["D"] for row in history], dtype=float)
    D_target_vals = np.array([row["D_target"] for row in history], dtype=float)
    D_low = np.array([row["D_low"] for row in history], dtype=float)
    D_high = np.array([row["D_high"] for row in history], dtype=float)

    T = np.array([row["T_ir"] for row in history], dtype=float)
    T_target_vals = np.array([row["T_target"] for row in history], dtype=float)

    violation = np.logical_or(D < D_low, D > D_high)

    speeds = np.array([row["v"] for row in history], dtype=float)
    nozzles = np.array([row["T_nozzle"] for row in history], dtype=float)
    u = np.column_stack([speeds, nozzles])
    tv_u = float(np.sum(np.linalg.norm(np.diff(u, axis=0), axis=1))) if len(u) > 1 else 0.0

    total_time = float(np.sum([row["dt"] for row in history]))

    return {
        "mean_abs_T_error": float(np.mean(np.abs(T - T_target_vals))),
        "mean_abs_D_error": float(np.mean(np.abs(D - D_target_vals))),
        "thermal_window_violation_rate": float(np.mean(violation)),
        "total_time_s": total_time,
        "TV_u": tv_u,
    }


def _plant_step(
    T: float,
    u: Sequence[float],
    alpha_deg: float,
    model: ThermalModelParams,
    cfg: MPCConfig,
    rng: np.random.Generator,
    noise_std: float = 0.7,
) -> Tuple[float, float, float, float]:
    """Advance the true plant one decision step; return T_meas, D_layer, dt, T_true."""
    T_next_true, D_layer, dt = analytical_layer_step(
        T=T,
        u=u,
        alpha_deg=alpha_deg,
        model=model,
        cfg=cfg,
        n_substeps=cfg.n_substeps,
    )
    T_meas = T_next_true + float(rng.normal(0.0, noise_std))
    return float(T_meas), float(D_layer), float(dt), float(T_next_true)


def _history_row(
    k: int,
    alpha_deg: float,
    T_meas: float,
    D_layer: float,
    u: Sequence[float],
    dt: float,
    **extra: float,
) -> Dict[str, float]:
    D_low, D_high = dose_window(alpha_deg)
    row: Dict[str, float] = {
        "k": float(k),
        "alpha_deg": float(alpha_deg),
        "T_ir": float(T_meas),
        "T_target": float(temperature_target(alpha_deg)),
        "D": float(D_layer),
        "D_target": float(dose_target(alpha_deg)),
        "D_low": float(D_low),
        "D_high": float(D_high),
        "v": float(u[0]),
        "T_nozzle": float(u[1]),
        "dt": float(dt),
    }
    row.update({key: float(val) for key, val in extra.items()})
    return row


def simulate_closed_loop_mpc(
    n_segments: int = 50,
    alpha_start: float = 45.0,
    alpha_end: float = 65.0,
    seed: int = 3,
) -> List[Dict[str, float]]:
    """Simulate MPC on a ramped overhang coupon with IR measurement noise."""
    rng = np.random.default_rng(seed)

    model_true = ThermalModelParams(
        T_inf=25.0,
        T_nozzle=215.0,
        T_g=60.0,
        beta=0.36,
        a_v=0.16,
        tau0=8.5,
        a_fan=2.2,
        a_alpha=0.5,
    )

    model_mpc = ThermalModelParams(
        T_inf=25.0,
        T_nozzle=215.0,
        T_g=60.0,
        beta=0.35,
        a_v=0.15,
        tau0=8.0,
        a_fan=2.0,
        a_alpha=0.4,
    )

    cfg = MPCConfig()
    alphas = np.linspace(alpha_start, alpha_end, n_segments)

    state = ControllerState(
        T=90.0,
        D=0.0,
        u_prev=np.array([100.0, 215.0], dtype=float),
    )

    history: List[Dict[str, float]] = []

    for k in range(n_segments):
        future_alphas = alphas[k : k + cfg.horizon]
        u_apply, info = solve_mpc(
            state=state,
            future_alphas=future_alphas,
            model=model_mpc,
            cfg=cfg,
        )

        T_meas, D_layer, dt, _ = _plant_step(
            state.T,
            u_apply,
            float(alphas[k]),
            model_true,
            cfg,
            rng,
        )

        history.append(
            _history_row(
                k,
                float(alphas[k]),
                T_meas,
                D_layer,
                u_apply,
                dt,
                mpc_cost=float(info["cost"]) if np.isfinite(info["cost"]) else float("nan"),
                mpc_success=float(bool(info["success"])),
            )
        )

        state = ControllerState(
            T=T_meas,
            D=D_layer,
            u_prev=np.array(u_apply, dtype=float),
        )

    return history


def simulate_fixed_baseline(
    n_segments: int = 50,
    alpha_start: float = 45.0,
    alpha_end: float = 65.0,
    u_fixed: Sequence[float] | None = None,
    seed: int = 4,
) -> List[Dict[str, float]]:
    """Simulate a fixed slicer setting baseline (outline §8.1)."""
    rng = np.random.default_rng(seed)
    model_true = ThermalModelParams()
    cfg = MPCConfig()
    alphas = np.linspace(alpha_start, alpha_end, n_segments)

    if u_fixed is None:
        # Use an in-bounds conventional point for fair offline comparison.
        u_fixed = (100.0, 215.0)

    T = 90.0
    history: List[Dict[str, float]] = []

    for k in range(n_segments):
        T_meas, D_layer, dt, _ = _plant_step(
            T,
            u_fixed,
            float(alphas[k]),
            model_true,
            cfg,
            rng,
        )
        history.append(_history_row(k, float(alphas[k]), T_meas, D_layer, u_fixed, dt))
        T = T_meas

    return history


def short_horizon_prediction_error(
    rows: List[Dict[str, float]],
    model: ThermalModelParams,
    cfg: MPCConfig,
    horizon: int = 3,
) -> float:
    """
    Mean absolute short-horizon prediction error e_N (outline §5.4).

    For each starting index k, roll the model forward N steps with recorded
    controls and compare to measured temperatures.
    """
    if len(rows) <= horizon:
        return float("nan")

    errors: List[float] = []
    for k in range(len(rows) - horizon):
        T = float(rows[k]["T_ir"] if "T_ir" in rows[k] else rows[k]["T_k"])
        for i in range(horizon):
            row = rows[k + i]
            alpha = float(row.get("alpha_deg", row.get("print_angle_deg", 60.0)))
            T_nozzle = float(
                row.get(
                    "T_nozzle",
                    row.get(
                        "nozzle_setpoint_c",
                        row.get("hotend_target", model.T_nozzle),
                    ),
                )
            )
            u = [
                float(row.get("v", row.get("print_speed_mms", 100.0))),
                T_nozzle,
            ]
            T, _, _ = analytical_layer_step(
                T,
                u,
                alpha,
                model,
                cfg,
                n_substeps=cfg.n_substeps,
            )
            measured = float(
                rows[k + i + 1].get(
                    "T_ir",
                    rows[k + i + 1].get("T_next_measured", rows[k + i + 1].get("T_k", T)),
                )
            )
            errors.append(abs(T - measured))

    return float(np.mean(errors)) if errors else float("nan")
