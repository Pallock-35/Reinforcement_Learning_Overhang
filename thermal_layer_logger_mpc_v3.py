#!/usr/bin/env python3
"""Layer-triggered conservative dual-input MPC for one overhang angle per run.

The controller varies only print-speed reference and nozzle setpoint. Fan is
always 100%, dwell is always zero, and thermal dose is logged but not optimized.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from baseline_controllers import fixed_control
from mpc_controller_v3 import ControllerState, MPCConfig, solve_mpc
from printer_control import (
    BASE_FLOW_PERCENT,
    apply_control_vector,
    clear_thermal_control_status,
    publish_thermal_control_status,
    read_thermal_summary,
)
from rl_layer_logger import (
    DEFAULT_OCTOPRINT_URL,
    LayerChangeWatcher,
    SPEED_SETTLE_TIME,
    load_octoprint_api_key,
    print_run_dir,
    print_run_folder_name,
    read_printer_temps,
    test_connections,
    write_print_run_metadata,
)
from thermal_dose import ThermalDoseTracker
from thermal_model import ThermalModelParams, model_params_from_dict
from thermal_targets import dose_target, dose_window, temperature_target
from thermal_stream_sampling import (
    STREAM_INTERVAL_S,
    create_stream_log_file,
    sample_stream_tick,
    stream_until_layer,
    write_stream_row,
)

PRINT_LOG_FILENAME = "log.csv"
N_PRINTS = 1
FAN_PERCENT_FIXED = 100.0
DWELL_S_FIXED = 0.0

V3_TARGET_C = 44.5
V3_T_DEADBAND_C = 0.75
V3_NOMINAL_SPEED_MMS = 100.0
V3_SPEED_MIN_MMS = 90.0
V3_SPEED_MAX_MMS = 110.0
V3_MAX_SPEED_STEP_MMS = 5.0
V3_INITIAL_NOZZLE_C = 210.0
V3_NOZZLE_MIN_C = 205.0
V3_NOZZLE_MAX_C = 215.0
V3_MAX_NOZZLE_STEP_C = 1.5
V3_WARMUP_LAYERS = 10
V3_CONTROL_MEDIAN_WINDOW = 3

THERMAL_LOG_COLUMNS = [
    "print_number",
    "print_angle_deg",
    "control_mode",
    "layer",
    "timestamp",
    "layer_dt_s",
    "print_speed_mms",
    "flow_percent",
    "fan_percent",
    "dwell_s",
    "nozzle_setpoint_c",
    "T_k",
    "D_k",
    "D_min",
    "D_max",
    "D_star",
    "T_star",
    "dT_dt",
    "roi_max",
    "roi_mean",
    "roi_std",
    "roi_min",
    "frame_max",
    "frame_mean",
    "hotend_actual",
    "hotend_target",
    "bed_actual",
    "bed_target",
    "new_roi_mean",
    "new_T_k",
    "T_control",
    "predicted_next_T_c",
    "proposed_speed_mms",
    "proposed_nozzle_c",
    "controller_phase",
    "mpc_cost",
    "mpc_success",
    "control_source",
    "stream_samples",
]


def create_log_file(
    print_number: int,
    *,
    print_angle_deg: float,
    control_mode: str,
) -> tuple[Any, Any, str, Any, Any, str]:
    run_dir = print_run_dir(
        print_number,
        control_mode=control_mode,
        print_angle_deg=print_angle_deg,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    filename = str(run_dir / PRINT_LOG_FILENAME)
    handle = open(filename, "w", newline="", encoding="utf-8")
    writer = csv.writer(handle)
    writer.writerow(THERMAL_LOG_COLUMNS)
    handle.flush()
    stream_handle, stream_writer, stream_filename = create_stream_log_file(str(run_dir))
    print(f"  Logging to: {filename}")
    print(f"  Stream log: {stream_filename} ({STREAM_INTERVAL_S:g} Hz between layer decisions)")
    return handle, writer, filename, stream_handle, stream_writer, stream_filename


def load_model_params(path: Path) -> ThermalModelParams:
    if not path.exists():
        raise FileNotFoundError(f"Thermal model file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return model_params_from_dict(payload)


def select_control(
    control_mode: str,
    *,
    alpha_deg: float,
    mpc_state: ControllerState,
    model: ThermalModelParams,
    cfg: MPCConfig,
    print_speed_mms: float,
    nozzle_temp_c: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    meta: dict[str, Any] = {
        "mpc_cost": "",
        "mpc_success": "",
        "control_source": control_mode,
        "predicted_next_T_c": "",
        "proposed_speed_mms": "",
        "proposed_nozzle_c": "",
    }

    if control_mode == "fixed":
        return fixed_control(print_speed_mms, nozzle_temp_c, cfg), meta

    if control_mode == "mpc":
        u, info = solve_mpc(
            state=mpc_state,
            alpha_deg=alpha_deg,
            model=model,
            cfg=cfg,
        )
        meta["mpc_cost"] = (
            round(float(info["cost"]), 4)
            if np.isfinite(info["cost"])
            else ""
        )
        meta["mpc_success"] = info["success"]
        meta["predicted_next_T_c"] = round(float(info["predicted_next_T_c"]), 3)
        meta["proposed_speed_mms"] = round(float(info["proposed_speed_mms"]), 3)
        meta["proposed_nozzle_c"] = round(float(info["proposed_nozzle_c"]), 3)
        return np.asarray(u, dtype=float), meta

    raise ValueError(f"Unknown control_mode: {control_mode}")


def _hard_limit_action(
    action: np.ndarray,
    previous: np.ndarray,
    cfg: MPCConfig,
) -> np.ndarray:
    """Second independent safety layer in the live logger."""
    action = np.asarray(action, dtype=float).copy()
    previous = np.asarray(previous, dtype=float).copy()

    action[0] = float(
        np.clip(
            action[0],
            max(cfg.v_min, previous[0] - cfg.dv_max),
            min(cfg.v_max, previous[0] + cfg.dv_max),
        )
    )
    action[1] = float(
        np.clip(
            action[1],
            max(cfg.T_nozzle_min, previous[1] - cfg.dT_nozzle_max),
            min(cfg.T_nozzle_max, previous[1] + cfg.dT_nozzle_max),
        )
    )
    return action


def run_one_print(
    print_number: int,
    octoprint_url: str,
    api_key: str,
    *,
    print_angle_deg: float,
    control_mode: str,
    print_speed_mms: float,
    nozzle_temp_c: float,
    model: ThermalModelParams,
    cfg: MPCConfig,
    warmup_layers: int,
    dry_run: bool = False,
) -> str:
    watcher = LayerChangeWatcher(octoprint_url, api_key)
    watcher.start()

    run_folder = print_run_folder_name(
        print_number,
        control_mode=control_mode,
        print_angle_deg=print_angle_deg,
    )
    meta_path = write_print_run_metadata(
        print_number,
        print_angle_deg=print_angle_deg,
        control_mode=control_mode,
    )

    print(f"\n{'='*78}")
    print(
        f"  PRINT #{print_number:>3} | one angle={print_angle_deg:.1f} deg | "
        f"mode={control_mode} | {datetime.now().strftime('%H:%M:%S')}"
    )
    print(f"  Run folder: {run_folder}/")
    print(f"  Run meta: {meta_path}")
    print(f"  Nominal: speed={print_speed_mms:.0f} mm/s nozzle={nozzle_temp_c:.0f} C fan=100%")
    print(f"{'='*78}")
    print(
        f"{'Layer':>6} {'T_ctl':>7} {'D_k':>7} {'Speed':>8} "
        f"{'Nozzle':>8} {'Source':>14}"
    )
    print("-" * 78)

    log_file, log_writer, log_filename, stream_file, stream_writer, stream_filename = create_log_file(
        print_number,
        print_angle_deg=print_angle_deg,
        control_mode=control_mode,
    )

    config_path = Path(log_filename).parent / "controller_config_v4.json"
    config_path.write_text(
        json.dumps(
            {
                "version": "dual-input-mpc-v4-empirical",
                "prediction_model": (
                    "T_next = -34.23 + 0.754*T_current "
                    "+ 0.0513*speed_mms "
                    "+ 0.1889*nozzle_target_c"
                ),
                "one_angle_per_run": True,
                "print_angle_deg": print_angle_deg,
                "temperature_signal": "3-layer median of roi_mean",
                "temperature_target_c": cfg.T_target_c,
                "temperature_deadband_c": cfg.T_deadband_c,
                "speed_nominal_mms": cfg.v_nominal,
                "speed_bounds_mms": [cfg.v_min, cfg.v_max],
                "max_speed_step_mms": cfg.dv_max,
                "nozzle_nominal_c": cfg.T_nozzle_nominal,
                "nozzle_bounds_c": [cfg.T_nozzle_min, cfg.T_nozzle_max],
                "max_nozzle_step_c": cfg.dT_nozzle_max,
                "warmup_layers": warmup_layers,
                "fan_percent": FAN_PERCENT_FIXED,
                "dwell_s": DWELL_S_FIXED,
                "dose_objective_enabled": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  Controller config: {config_path}")

    dose_tracker = ThermalDoseTracker(T_g=model.T_g)
    if control_mode == "mpc":
        u_prev = np.array(
            [
                float(np.clip(print_speed_mms, cfg.v_min, cfg.v_max)),
                float(np.clip(nozzle_temp_c, cfg.T_nozzle_min, cfg.T_nozzle_max)),
            ],
            dtype=float,
        )
    else:
        u_prev = fixed_control(print_speed_mms, nozzle_temp_c, cfg)

    if not dry_run:
        apply_control_vector(u_prev, flow_percent=BASE_FLOW_PERCENT)

    layers_completed = 0
    last_decision_layer = 0
    decision_monotonic = time.monotonic()
    speed_mms = float(u_prev[0])
    nozzle_setpoint_c = float(u_prev[1])
    temperature_history: deque[float] = deque(maxlen=V3_CONTROL_MEDIAN_WINDOW)

    try:
        watcher.wait_for_print_start()

        while True:
            layer, stream_state = stream_until_layer(
                watcher,
                dose_tracker=dose_tracker,
                stream_writer=stream_writer,
                stream_handle=stream_file,
                read_thermal_summary=read_thermal_summary,
                read_printer_temps=read_printer_temps,
                print_number=print_number,
                print_angle_deg=print_angle_deg,
                control_mode=control_mode,
                last_decision_layer=last_decision_layer,
                decision_monotonic=decision_monotonic,
                print_speed_mms=speed_mms,
                nozzle_setpoint_c=nozzle_setpoint_c,
            )
            if layer is None:
                print("\n  Print finished - ending run.")
                break

            if stream_state.samples_since_decision == 0:
                boundary_dt = max(time.monotonic() - decision_monotonic, 0.001)
                stream_state = sample_stream_tick(
                    dose_tracker,
                    boundary_dt,
                    read_thermal_summary,
                )
                stream_state.samples_since_decision = 1
                try:
                    boundary_temps = read_printer_temps()
                except Exception:
                    boundary_temps = {}
                write_stream_row(
                    stream_writer,
                    print_number=print_number,
                    print_angle_deg=print_angle_deg,
                    control_mode=control_mode,
                    last_decision_layer=last_decision_layer,
                    sample_dt_s=boundary_dt,
                    elapsed_since_decision_s=boundary_dt,
                    stream_state=stream_state,
                    print_speed_mms=speed_mms,
                    nozzle_setpoint_c=nozzle_setpoint_c,
                    temps=boundary_temps,
                )

            layers_completed += 1
            layer_dt = time.monotonic() - decision_monotonic
            T_k = float(stream_state.T_k)
            D_k = float(stream_state.D_k)
            dT_dt = float(stream_state.dT_dt)
            thermal = stream_state.thermal or read_thermal_summary()

            D_star = dose_target(print_angle_deg)
            D_min, D_max = dose_window(print_angle_deg)
            T_star = cfg.T_target_c if control_mode == "mpc" else temperature_target(print_angle_deg)

            temperature_history.append(T_k)
            T_control = float(np.median(np.asarray(temperature_history, dtype=float)))

            if control_mode == "mpc" and layer <= warmup_layers:
                u_apply = np.array([cfg.v_nominal, cfg.T_nozzle_nominal], dtype=float)
                ctrl_meta = {
                    "mpc_cost": "",
                    "mpc_success": "",
                    "control_source": "warmup_fixed",
                    "predicted_next_T_c": "",
                    "proposed_speed_mms": cfg.v_nominal,
                    "proposed_nozzle_c": cfg.T_nozzle_nominal,
                }
                controller_phase = "warmup_fixed"
            else:
                mpc_state = ControllerState(T=T_control, u_prev=u_prev.copy())
                u_apply, ctrl_meta = select_control(
                    control_mode,
                    alpha_deg=print_angle_deg,
                    mpc_state=mpc_state,
                    model=model,
                    cfg=cfg,
                    print_speed_mms=print_speed_mms,
                    nozzle_temp_c=nozzle_temp_c,
                )
                controller_phase = control_mode

            if control_mode == "mpc":
                u_apply = _hard_limit_action(u_apply, u_prev, cfg)

            speed_mms = float(u_apply[0])
            nozzle_setpoint_c = float(u_apply[1])

            if not dry_run:
                apply_control_vector(u_apply, flow_percent=BASE_FLOW_PERCENT)
                publish_thermal_control_status(
                    control_mode=control_mode,
                    speed_mms=speed_mms,
                    nozzle_temp_c=nozzle_setpoint_c,
                    print_number=print_number,
                    layer=layer,
                    T_k=T_k,
                    D_k=D_k,
                    mpc_cost=ctrl_meta.get("mpc_cost") if ctrl_meta.get("mpc_cost") != "" else None,
                    mpc_success=ctrl_meta.get("mpc_success") if ctrl_meta.get("mpc_success") != "" else None,
                )

            time.sleep(SPEED_SETTLE_TIME)
            new_thermal = read_thermal_summary()
            new_temps = read_printer_temps()
            new_T_k = float(new_thermal["roi_mean"])

            print(
                f"{layer:>6} {T_control:>6.1f}C {D_k:>7.1f} "
                f"{speed_mms:>6.1f} {nozzle_setpoint_c:>7.1f}C "
                f"{ctrl_meta.get('control_source', control_mode):>14}"
            )

            log_writer.writerow([
                print_number,
                round(print_angle_deg, 2),
                control_mode,
                layer,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                round(layer_dt, 3),
                round(speed_mms, 2),
                round(BASE_FLOW_PERCENT, 2),
                round(FAN_PERCENT_FIXED, 2),
                round(DWELL_S_FIXED, 3),
                round(nozzle_setpoint_c, 2),
                round(T_k, 2),
                round(D_k, 2),
                round(D_min, 2),
                round(D_max, 2),
                round(D_star, 2),
                round(T_star, 2),
                round(dT_dt, 4),
                round(thermal.get("roi_max", float("nan")), 2),
                round(thermal.get("roi_mean", float("nan")), 2),
                round(thermal.get("roi_std", float("nan")), 2),
                round(thermal.get("roi_min", float("nan")), 2),
                round(thermal.get("frame_max", float("nan")), 2),
                round(thermal.get("frame_mean", float("nan")), 2),
                new_temps.get("hotend_actual", ""),
                new_temps.get("hotend_target", ""),
                new_temps.get("bed_actual", ""),
                new_temps.get("bed_target", ""),
                round(new_thermal["roi_mean"], 2),
                round(new_T_k, 2),
                round(T_control, 2),
                ctrl_meta.get("predicted_next_T_c", ""),
                ctrl_meta.get("proposed_speed_mms", ""),
                ctrl_meta.get("proposed_nozzle_c", ""),
                controller_phase,
                ctrl_meta.get("mpc_cost", ""),
                ctrl_meta.get("mpc_success", ""),
                ctrl_meta.get("control_source", control_mode),
                stream_state.samples_since_decision,
            ])
            log_file.flush()
            stream_file.flush()
            u_prev = np.asarray(u_apply, dtype=float).copy()
            last_decision_layer = layer
            decision_monotonic = time.monotonic()
            dose_tracker.reset()

    except KeyboardInterrupt:
        print("\n  Run interrupted.")
    finally:
        watcher.stop()
        log_file.close()
        stream_file.close()
        if not dry_run:
            clear_thermal_control_status()

    print(f"\n  Print #{print_number} done - {layers_completed} layer(s) logged")
    print(f"  Layer log: {log_filename}")
    print(f"  Stream log: {stream_filename}")
    return log_filename


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Conservative dual-input IR-informed MPC"
    )
    parser.add_argument(
        "--control-mode",
        choices=("fixed", "mpc"),
        default="mpc",
        help="Controller mode (default: mpc)",
    )
    parser.add_argument("--print-speed", type=float, default=V3_NOMINAL_SPEED_MMS)
    parser.add_argument("--nozzle-temp", type=float, default=V3_INITIAL_NOZZLE_C)
    parser.add_argument(
        "--print-angle",
        type=float,
        required=True,
        help="One overhang angle for this run; use a separate run for each angle",
    )
    parser.add_argument(
        "--model-params",
        type=Path,
        default=Path("models/thermal_fit_boundary.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prints", type=int, default=N_PRINTS)
    parser.add_argument("--temperature-target", type=float, default=V3_TARGET_C)
    parser.add_argument("--temperature-deadband", type=float, default=V3_T_DEADBAND_C)
    parser.add_argument("--speed-min", type=float, default=V3_SPEED_MIN_MMS)
    parser.add_argument("--speed-max", type=float, default=V3_SPEED_MAX_MMS)
    parser.add_argument("--max-speed-step", type=float, default=V3_MAX_SPEED_STEP_MMS)
    parser.add_argument("--nozzle-min", type=float, default=V3_NOZZLE_MIN_C)
    parser.add_argument("--nozzle-max", type=float, default=V3_NOZZLE_MAX_C)
    parser.add_argument("--max-nozzle-step", type=float, default=V3_MAX_NOZZLE_STEP_C)
    parser.add_argument("--warmup-layers", type=int, default=V3_WARMUP_LAYERS)
    args = parser.parse_args()

    if not 0.0 <= args.print_angle <= 90.0:
        print("--print-angle must be between 0 and 90 degrees.")
        return 1
    if args.prints != 1:
        print("MPC v4 commissioning is restricted to --prints 1.")
        return 1
    if args.speed_min >= args.speed_max:
        print("--speed-min must be lower than --speed-max.")
        return 1
    if args.nozzle_min >= args.nozzle_max:
        print("--nozzle-min must be lower than --nozzle-max.")
        return 1
    if not args.speed_min <= args.print_speed <= args.speed_max:
        print("--print-speed must lie inside the speed bounds.")
        return 1
    if not args.nozzle_min <= args.nozzle_temp <= args.nozzle_max:
        print("--nozzle-temp must lie inside the nozzle bounds.")
        return 1
    if args.max_speed_step <= 0 or args.max_nozzle_step <= 0:
        print("Rate limits must be positive.")
        return 1
    if not args.model_params.exists():
        print(f"ERROR: fitted model not found: {args.model_params}")
        return 1

    api_key = load_octoprint_api_key()
    if not api_key:
        print(
            "ERROR: OctoPrint API key is required for "
            "layer detection, including dry-run."
        )
        return 1

    print("\nThermal Layer Logger - empirical dual-input MPC v4")
    print(f"One angle this run: {args.print_angle:g} deg")
    print(f"Fan: 100% locked | Dwell: 0 locked | Dose objective: disabled")
    print(f"Primary feedback: 3-layer median of roi_mean ({STREAM_INTERVAL_S:g} Hz stream)")

    if not args.dry_run and not test_connections():
        return 1

    model = load_model_params(args.model_params)
    model = ThermalModelParams(
        T_inf=model.T_inf,
        T_nozzle=float(args.nozzle_temp),
        T_g=model.T_g,
        beta=model.beta,
        a_v=model.a_v,
        tau0=model.tau0,
        a_fan=model.a_fan,
        a_alpha=model.a_alpha,
    )

    cfg = MPCConfig(
        horizon=2,
        v_min=float(args.speed_min),
        v_max=float(args.speed_max),
        T_nozzle_min=float(args.nozzle_min),
        T_nozzle_max=float(args.nozzle_max),
        dv_max=float(args.max_speed_step),
        dT_nozzle_max=float(args.max_nozzle_step),
        T_target_c=float(args.temperature_target),
        T_deadband_c=float(args.temperature_deadband),
        v_nominal=float(args.print_speed),
        T_nozzle_nominal=float(args.nozzle_temp),
        w_T=1.0,
        w_du=0.50,
        w_u_nominal=0.25,
        w_rate_violation=20.0,
    )

    print("\nMPC V4 EMPIRICAL DUAL-INPUT CONFIGURATION")
    print("-" * 56)
    print(f"ROI target/deadband:   {cfg.T_target_c:.2f} +/- {cfg.T_deadband_c:.2f} C")
    print(f"Speed nominal/bounds:  {cfg.v_nominal:.1f} [{cfg.v_min:.1f}, {cfg.v_max:.1f}] mm/s")
    print(f"Max speed step:        +/-{cfg.dv_max:.1f} mm/s/layer")
    print(f"Nozzle nominal/bounds: {cfg.T_nozzle_nominal:.1f} [{cfg.T_nozzle_min:.1f}, {cfg.T_nozzle_max:.1f}] C")
    print(f"Max nozzle step:       +/-{cfg.dT_nozzle_max:.1f} C/layer")
    print(f"Warmup layers:         {args.warmup_layers}")
    print("-" * 56)

    # Always obtain the active OctoPrint URL from the bridge,
    # including during dry-run.
    octoprint_url = (
        read_printer_temps().get("octoprint_url")
        or DEFAULT_OCTOPRINT_URL
    )

    print(f"Layer polling URL: {octoprint_url}")

    if not args.dry_run:
        input("Press Enter only after the single-angle print is running...")

    run_one_print(
        1,
        octoprint_url,
        api_key,
        print_angle_deg=float(args.print_angle),
        control_mode=args.control_mode,
        print_speed_mms=float(args.print_speed),
        nozzle_temp_c=float(args.nozzle_temp),
        model=model,
        cfg=cfg,
        warmup_layers=int(args.warmup_layers),
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
