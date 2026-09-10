"""
IR-informed MPC for layerwise thermal control — compatibility shim.

Core logic lives in thermal_targets, thermal_model, and mpc_controller.
"""

from __future__ import annotations

from mpc_controller import (
    ControllerState,
    MPCConfig,
    conservative_fallback,
    mpc_objective,
    short_horizon_prediction_error,
    simulate_closed_loop_mpc,
    simulate_fixed_baseline,
    soft_rate_violation,
    solve_mpc,
    summarize_history,
)
from printer_control import apply_controls as apply_printer_controls
from printer_control import read_ir_temperature as read_ir_temperature_from_sensor
from thermal_model import (
    ThermalModelParams,
    analytical_layer_step,
    analytical_thermal_step,
    fit_thermal_model,
    pack_params,
    unpack_params,
)
from thermal_targets import (
    dose_target,
    dose_window,
    temperature_bond_limit,
    temperature_sag_limit,
    temperature_target,
)

__all__ = [
    "ThermalModelParams",
    "MPCConfig",
    "ControllerState",
    "temperature_target",
    "temperature_bond_limit",
    "temperature_sag_limit",
    "dose_target",
    "dose_window",
    "analytical_thermal_step",
    "analytical_layer_step",
    "soft_rate_violation",
    "mpc_objective",
    "solve_mpc",
    "conservative_fallback",
    "pack_params",
    "unpack_params",
    "fit_thermal_model",
    "read_ir_temperature_from_sensor",
    "apply_printer_controls",
    "simulate_closed_loop_mpc",
    "simulate_fixed_baseline",
    "short_horizon_prediction_error",
    "summarize_history",
]


if __name__ == "__main__":
    mpc_history = simulate_closed_loop_mpc()
    fixed_history = simulate_fixed_baseline()

    print("\nFixed baseline summary")
    print(summarize_history(fixed_history))

    print("\nMPC summary")
    print(summarize_history(mpc_history))

    print("\nFirst five MPC actions")
    for row in mpc_history[:5]:
        print(
            f"k={int(row['k']):02d}, "
            f"alpha={row['alpha_deg']:.1f}, "
            f"T={row['T_ir']:.1f}, "
            f"D={row['D']:.1f}, "
            f"v={row['v']:.1f}, "
            f"T_nozzle={row['T_nozzle']:.1f}"
        )
