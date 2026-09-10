#!/usr/bin/env python3
"""
thermal_layer_logger.py
=======================
Layer-triggered IR-informed thermal control for overhang tower prints.

Control modes (--control-mode):
  fixed     — hold --print-speed and --nozzle-temp for the whole print
  mpc       — live SLSQP MPC on speed + nozzle (warm-started near V/T)
  mpc_temp  — speed fixed at V; MPC varies nozzle only

Fan is always 100% (M106 S255). Dwell is always 0 (no G4).
Control vector: u = [v_mms, T_nozzle_C].

Primary IR feedback: T_k = roi_mean (T_mean,k). Between layer decisions the logger
samples IR at 1 Hz into stream.csv and integrates per-layer D_k; the dose tracker
resets after each decision so D*(α)±ΔD stays on a layer scale.

Depends on dashboard_bridge.py for thermal reads and G-code actuation.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from baseline_controllers import (
    DEFAULT_NOZZLE_TEMP_C,
    DEFAULT_PRINT_SPEED_MMS,
    clip_control,
    fixed_control,
)
from mpc_controller import ControllerState, MPCConfig, solve_mpc
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
N_PRINTS = 20
FAN_PERCENT_FIXED = 100.0
DWELL_S_FIXED = 0.0


# ============================================================
# FIRST MPC TEMPERATURE-CONTROL COMMISSIONING SETTINGS
# ============================================================

COMMISSIONING_FIXED_SPEED_MMS = 100.0
COMMISSIONING_INITIAL_NOZZLE_C = 210.0

COMMISSIONING_NOZZLE_MIN_C = 200.0
COMMISSIONING_NOZZLE_MAX_C = 220.0

COMMISSIONING_T_TARGET_C = 46.6
COMMISSIONING_MAX_NOZZLE_STEP_C = 3.0

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


def load_model_params(path: Path | None) -> ThermalModelParams:
    if path is None or not path.exists():
        return ThermalModelParams()
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
    }

    if control_mode == "fixed":
        return fixed_control(print_speed_mms, nozzle_temp_c, cfg), meta

    if control_mode == "mpc":
        u, info = solve_mpc(
            mpc_state,
            [alpha_deg] * cfg.horizon,
            model,
            cfg,
        )
        meta["mpc_cost"] = round(float(info["cost"]), 4) if np.isfinite(info["cost"]) else ""
        meta["mpc_success"] = info["success"]
        return clip_control(u, cfg), meta

    if control_mode == "mpc_temp":
        # Hold print speed at V; optimize nozzle only.
        speed = float(np.clip(print_speed_mms, cfg.v_min, cfg.v_max))
        cfg_temp = replace(cfg, v_min=speed, v_max=speed)
        state = ControllerState(
            T=mpc_state.T,
            D=mpc_state.D,
            u_prev=np.array([speed, float(mpc_state.u_prev[1])], dtype=float),
        )
        u, info = solve_mpc(
            state,
            [alpha_deg] * cfg.horizon,
            model,
            cfg_temp,
        )
        u = np.array([speed, float(u[1])], dtype=float)
        meta["mpc_cost"] = round(float(info["cost"]), 4) if np.isfinite(info["cost"]) else ""
        meta["mpc_success"] = info["success"]
        return clip_control(u, cfg_temp), meta

    raise ValueError(f"Unknown control_mode: {control_mode}")


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

    print(f"\n{'='*72}")
    print(
        f"  PRINT #{print_number:>3}  |  angle={print_angle_deg:.1f}°  |  "
        f"mode={control_mode}  |  {datetime.now().strftime('%H:%M:%S')}"
    )
    print(f"  Run folder: {run_folder}/")
    print(f"  Run meta: {meta_path}")
    print(f"  Baseline: speed={print_speed_mms:.0f} mm/s  nozzle={nozzle_temp_c:.0f}C  fan=100%")
    print(f"{'='*72}")
    print(
        f"{'Layer':>6}  {'T_k':>7}  {'D_k':>7}  {'Speed':>7}  "
        f"{'Nozzle':>7}  {'Source':>12}"
    )
    print("-" * 72)

    log_file, log_writer, log_filename, stream_file, stream_writer, stream_filename = create_log_file(
        print_number,
        print_angle_deg=print_angle_deg,
        control_mode=control_mode,
    )

    if control_mode == "mpc_temp":
        controller_config_path = (
            Path(log_filename).parent
            / "controller_config.json"
        )

        controller_config = {
            "commissioning_run": True,
            "control_mode": control_mode,
            "print_angle_deg": float(
                print_angle_deg
            ),

            "temperature_signal": "roi_mean",
            "temperature_target_c": float(
                cfg.T_target_override_c
            ),

            "fixed_speed_mms": float(
                print_speed_mms
            ),
            "initial_nozzle_c": float(
                nozzle_temp_c
            ),

            "nozzle_min_c": float(
                cfg.T_nozzle_min
            ),
            "nozzle_max_c": float(
                cfg.T_nozzle_max
            ),
            "maximum_nozzle_step_c": float(
                cfg.dT_nozzle_max
            ),

            "fan_percent": float(
                FAN_PERCENT_FIXED
            ),
            "dwell_s": float(
                DWELL_S_FIXED
            ),

            "temperature_weight": float(
                cfg.w_T
            ),
            "dose_weight": float(
                cfg.w_D
            ),
            "dose_window_weight": float(
                cfg.w_window
            ),
            "temperature_constraint_weight": float(
                cfg.w_temp_constraint
            ),

            "offline_exposure_signal": "roi_mean",
            "offline_exposure_threshold_c": 45.0,

            "model_parameters": {
                "T_inf": float(model.T_inf),
                "T_nozzle": float(
                    model.T_nozzle
                ),
                "T_g": float(model.T_g),
                "beta": float(model.beta),
                "a_v": float(model.a_v),
                "tau0": float(model.tau0),
                "a_fan": float(model.a_fan),
                "a_alpha": float(
                    model.a_alpha
                ),
            },
        }

        controller_config_path.write_text(
            json.dumps(
                controller_config,
                indent=4,
            ),
            encoding="utf-8",
        )

        print(
            f"  Controller config: "
            f"{controller_config_path}"
        )

    dose_tracker = ThermalDoseTracker(T_g=model.T_g)
    # MPC modes start near the CLI V/T; fixed holds them for the whole print.
    if control_mode in ("mpc", "mpc_temp"):
        speed0 = float(np.clip(print_speed_mms, cfg.v_min, cfg.v_max))
        nozzle0 = float(np.clip(nozzle_temp_c, cfg.T_nozzle_min, cfg.T_nozzle_max))
        u_prev = np.array([speed0, nozzle0], dtype=float)
    else:
        u_prev = fixed_control(print_speed_mms, nozzle_temp_c, cfg)

    if not dry_run:
        apply_control_vector(u_prev, flow_percent=BASE_FLOW_PERCENT)

    layers_completed = 0
    last_decision_layer = 0
    decision_monotonic = time.monotonic()
    speed_mms = float(u_prev[0])
    nozzle_setpoint_c = float(u_prev[1])

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
                print("\n  Print finished — ending run.")
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

            # Outline §6.6: T_k <- T_IR,k and D_k from latest IR stream at layer boundary.
            T_k = float(stream_state.T_k)
            D_k = float(stream_state.D_k)
            dT_dt = float(stream_state.dT_dt)
            thermal = stream_state.thermal or read_thermal_summary()

            D_star = dose_target(print_angle_deg)
            D_min, D_max = dose_window(print_angle_deg)

            if cfg.T_target_override_c is not None:
                T_star = float(cfg.T_target_override_c)
            else:
                T_star = temperature_target(
                    print_angle_deg
                )

            mpc_state = ControllerState(T=T_k, D=D_k, u_prev=u_prev.copy())
            u_apply, ctrl_meta = select_control(
                control_mode,
                alpha_deg=print_angle_deg,
                mpc_state=mpc_state,
                model=model,
                cfg=cfg,
                print_speed_mms=print_speed_mms,
                nozzle_temp_c=nozzle_temp_c,
            )

            u_apply = np.asarray(
                u_apply,
                dtype=float,
            ).copy()


            # ------------------------------------------------------------
            # HARD APPLIED-COMMAND LIMITS FOR TEMPERATURE-ONLY MPC
            # ------------------------------------------------------------
            if control_mode == "mpc_temp":

                # Hold the print-speed reference exactly fixed.
                fixed_speed_mms = float(
                    np.clip(
                        print_speed_mms,
                        cfg.v_min,
                        cfg.v_max,
                    )
                )

                u_apply[0] = fixed_speed_mms

                # Absolute nozzle-temperature limits.
                proposed_nozzle_c = float(
                    np.clip(
                        u_apply[1],
                        cfg.T_nozzle_min,
                        cfg.T_nozzle_max,
                    )
                )

                previous_nozzle_c = float(u_prev[1])
                maximum_step_c = float(
                    cfg.dT_nozzle_max
                )

                # Maximum allowed change relative to the last
                # command that was actually applied.
                rate_limited_min_c = max(
                    cfg.T_nozzle_min,
                    previous_nozzle_c - maximum_step_c,
                )

                rate_limited_max_c = min(
                    cfg.T_nozzle_max,
                    previous_nozzle_c + maximum_step_c,
                )

                u_apply[1] = float(
                    np.clip(
                        proposed_nozzle_c,
                        rate_limited_min_c,
                        rate_limited_max_c,
                    )
                )


            speed_mms = float(u_apply[0])
            nozzle_setpoint_c = float(u_apply[1])

            if not dry_run:
                apply_control_vector(
                    u_apply,
                    flow_percent=BASE_FLOW_PERCENT,
                )
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
                f"{layer:>6}  {T_k:>6.1f}C  {D_k:>7.1f}  {speed_mms:>5.0f}mm/s  "
                f"{nozzle_setpoint_c:>5.0f}C  "
                f"{ctrl_meta.get('control_source', control_mode):>12}  "
                f"(stream n={stream_state.samples_since_decision})"
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
                new_temps["hotend_actual"],
                new_temps["hotend_target"],
                new_temps["bed_actual"],
                new_temps["bed_target"],
                round(new_thermal["roi_mean"], 2),
                round(new_T_k, 2),
                ctrl_meta.get("mpc_cost", ""),
                ctrl_meta.get("mpc_success", ""),
                ctrl_meta.get("control_source", control_mode),
                stream_state.samples_since_decision,
            ])
            log_file.flush()
            stream_file.flush()
            u_prev = u_apply.copy()
            last_decision_layer = layer
            decision_monotonic = time.monotonic()
            # Per-layer dose: D_k is the integral over the completed layer only.
            dose_tracker.reset()

    except KeyboardInterrupt:
        print("\n  Run interrupted.")
    finally:
        watcher.stop()
        log_file.close()
        stream_file.close()
        if not dry_run:
            clear_thermal_control_status()

    print(f"\n  Print #{print_number} done — {layers_completed} layer(s) logged")
    print(f"  Layer log: {log_filename}")
    print(f"  Stream log: {stream_filename}")
    return log_filename


def main() -> int:
    parser = argparse.ArgumentParser(description="IR-informed thermal layer logger")
    parser.add_argument(
        "--control-mode",
        choices=("fixed", "mpc", "mpc_temp"),
        default="fixed",
        help="Controller mode (default: fixed)",
    )
    parser.add_argument(
        "--print-speed",
        type=float,
        default=DEFAULT_PRINT_SPEED_MMS,
        metavar="MMS",
        help=f"Print speed mm/s (default {DEFAULT_PRINT_SPEED_MMS:g})",
    )
    parser.add_argument(
        "--nozzle-temp",
        type=float,
        default=DEFAULT_NOZZLE_TEMP_C,
        metavar="C",
        help=f"Nozzle setpoint deg C (default {DEFAULT_NOZZLE_TEMP_C:g})",
    )
    parser.add_argument(
        "--print-angle",
        type=float,
        metavar="DEG",
        help="Overhang angle from vertical (prompted if omitted)",
    )
    parser.add_argument(
        "--model-params",
        type=Path,
        default=Path("models/thermal_fit.json"),
        help="JSON thermal model for MPC",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log and compute controls without sending G-code",
    )
    parser.add_argument(
        "--prints",
        type=int,
        default=N_PRINTS,
        help=f"Number of prints in session (default {N_PRINTS})",
    )
    parser.add_argument(
        "--temperature-target",
        type=float,
        default=COMMISSIONING_T_TARGET_C,
        metavar="C",
        help=(
            "ROI-mean temperature target for mpc_temp "
            f"(default {COMMISSIONING_T_TARGET_C:g} C)"
        ),
    )

    parser.add_argument(
        "--nozzle-min",
        type=float,
        default=COMMISSIONING_NOZZLE_MIN_C,
        metavar="C",
        help=(
            "Minimum nozzle temperature for mpc_temp "
            f"(default {COMMISSIONING_NOZZLE_MIN_C:g} C)"
        ),
    )

    parser.add_argument(
        "--nozzle-max",
        type=float,
        default=COMMISSIONING_NOZZLE_MAX_C,
        metavar="C",
        help=(
            "Maximum nozzle temperature for mpc_temp "
            f"(default {COMMISSIONING_NOZZLE_MAX_C:g} C)"
        ),
    )

    parser.add_argument(
        "--max-nozzle-step",
        type=float,
        default=COMMISSIONING_MAX_NOZZLE_STEP_C,
        metavar="C",
        help=(
            "Maximum applied nozzle-temperature change per layer "
            f"(default {COMMISSIONING_MAX_NOZZLE_STEP_C:g} C)"
        ),
    )

    args = parser.parse_args()

    if args.print_angle is not None and (
        args.print_angle < 0.0 or args.print_angle > 90.0
    ):
        print("--print-angle must be between 0 and 90 degrees.")
        return 1

    if args.print_speed <= 0.0:
        print("--print-speed must be positive.")
        return 1

    if args.control_mode == "mpc_temp":

        if args.nozzle_min >= args.nozzle_max:
            print("--nozzle-min must be lower than --nozzle-max.")
            return 1

        if not (
            args.nozzle_min
            <= args.nozzle_temp
            <= args.nozzle_max
        ):
            print(
                "--nozzle-temp must be inside the selected "
                "nozzle-temperature bounds."
            )
            return 1

        if args.max_nozzle_step <= 0.0:
            print("--max-nozzle-step must be positive.")
            return 1

        if not np.isfinite(args.temperature_target):
            print("--temperature-target must be finite.")
            return 1

    api_key = load_octoprint_api_key()
    if not api_key and not args.dry_run:
        print("Set PRUSASAM_OCTOPRINT_API_KEY or create api_key.py for layer detection.")
        return 1

    print("\nThermal Layer Logger — IR-informed MPC overhang control")
    print(f"Control mode: {args.control_mode}")
    print(f"Print speed:  {args.print_speed:g} mm/s")
    print(f"Nozzle temp:  {args.nozzle_temp:g} C")
    print(f"Fan:          100% (locked)  |  Dwell: 0 (locked)")
    print(f"Primary T_k: roi_mean (layer decisions use latest {STREAM_INTERVAL_S:g} Hz stream)\n")

    if not args.dry_run and not test_connections():
        return 1

    model_file_exists = (
        args.model_params is not None
        and args.model_params.exists()
    )

    if (
        args.control_mode in ("mpc", "mpc_temp")
        and not model_file_exists
    ):
        if args.dry_run:
            print(
                "\nWARNING: thermal model file was not found:\n"
                f"  {args.model_params}\n"
                "The dry run will use the default model parameters.\n"
                "Do not proceed to a physical MPC run based only "
                "on an unvalidated default model.\n"
            )
        else:
            print(
                "\nERROR: physical MPC run blocked because the "
                "thermal-model file was not found:\n"
                f"  {args.model_params}\n"
                "Provide a validated model file or run with "
                "--dry-run first."
            )
            return 1
    

    model = load_model_params(args.model_params)
    # Seed model default nozzle from CLI so deposit term stays consistent offline.
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
    cfg = MPCConfig()

    if args.control_mode == "mpc_temp":
        cfg = replace(
            cfg,

            # Hard SLSQP nozzle bounds
            T_nozzle_min=float(args.nozzle_min),
            T_nozzle_max=float(args.nozzle_max),

            # Used by the soft optimizer penalty and the
            # hard applied-command clamp below
            dT_nozzle_max=float(args.max_nozzle_step),

            # Temperature-only commissioning objective
            w_T=1.0,
            w_D=0.0,
            w_window=0.0,
            w_temp_constraint=0.0,

            # Measured reference from the conventional experiment
            T_target_override_c=float(
                args.temperature_target
            ),
        )

        print("\nMPC TEMPERATURE-CONTROL CONFIGURATION")
        print("-" * 52)
        print("Temperature signal:    roi_mean")
        print(
            f"Temperature target:    "
            f"{cfg.T_target_override_c:.2f} C"
        )
        print(
            f"Fixed speed:           "
            f"{args.print_speed:.1f} mm/s"
        )
        print(
            f"Initial nozzle:        "
            f"{args.nozzle_temp:.1f} C"
        )
        print(
            f"Nozzle bounds:         "
            f"{cfg.T_nozzle_min:.1f}-"
            f"{cfg.T_nozzle_max:.1f} C"
        )
        print(
            f"Maximum nozzle step:   "
            f"+/-{cfg.dT_nozzle_max:.1f} C/layer"
        )
        print("Fan:                   100%")
        print("Dose control:          disabled")
        print("45 C exposure:         offline diagnostic")
        print("-" * 52)

    octoprint_url = DEFAULT_OCTOPRINT_URL
    if not args.dry_run:
        octoprint_url = read_printer_temps()["octoprint_url"] or DEFAULT_OCTOPRINT_URL

    default_angle = args.print_angle
    for print_number in range(1, args.prints + 1):
        if default_angle is None:
            default_angle = _prompt_print_angle()
        print_angle = default_angle
        if args.print_angle is None:
            revise = input(
                f"  Use angle {print_angle:.1f}° for print #{print_number}? (Enter=yes / number): "
            ).strip()
            if revise:
                try:
                    print_angle = float(revise)
                except ValueError:
                    pass
            default_angle = print_angle

        if not args.dry_run:
            input("  Press Enter when the print is running...")
        run_one_print(
            print_number,
            octoprint_url,
            api_key,
            print_angle_deg=print_angle,
            control_mode=args.control_mode,
            print_speed_mms=float(args.print_speed),
            nozzle_temp_c=float(args.nozzle_temp),
            model=model,
            cfg=cfg,
            dry_run=args.dry_run,
        )
        if print_number < args.prints:
            answer = input(f"\n  Continue to print #{print_number + 1}? (y/n): ")
            if answer.strip().lower() != "y":
                break
    return 0


def _prompt_print_angle(suggested: float | None = None) -> float:
    hint = f" [{suggested:.1f}]" if suggested is not None else ""
    while True:
        answer = input(f"  Overhang angle from vertical (degrees){hint}: ").strip()
        if not answer and suggested is not None:
            return suggested
        try:
            angle = float(answer)
        except ValueError:
            print("  Enter a numeric angle (e.g. 60).")
            continue
        if 0.0 <= angle <= 90.0:
            return angle
        print("  Angle must be between 0 and 90 degrees.")


if __name__ == "__main__":
    raise SystemExit(main())
