"""Bridge-backed IR read and printer actuation (M220/M106/M104)."""

from __future__ import annotations

import os
import time
from typing import Any, Sequence

import numpy as np
import requests

from thermal_roi import summarize_deposit_roi

BRIDGE_URL = os.environ.get("PRUSASAM_BRIDGE_HOST", "http://127.0.0.1:8765")
BRIDGE_URL = (
    BRIDGE_URL if BRIDGE_URL.startswith("http") else f"http://{BRIDGE_URL}:8765"
).rstrip("/")

THERMAL_URL = f"{BRIDGE_URL}/api/thermal"
GCODE_URL = f"{BRIDGE_URL}/api/gcode"
THERMAL_CONTROL_STATUS_URL = f"{BRIDGE_URL}/api/thermal-control-status"

BASE_SPEED_MMS = 170.0
BASE_FLOW_PERCENT = 100.0
FAN_PWM_FULL = 255
NOZZLE_TEMP_MIN_C = 190.0
NOZZLE_TEMP_MAX_C = 230.0


def bridge_thermal_matrix() -> np.ndarray:
    response = requests.get(THERMAL_URL, timeout=5)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("ok"):
        raise RuntimeError(f"Thermal feed not ready: {payload.get('error', 'unknown')}")
    thermal = payload["thermal"]
    if not thermal.get("connected"):
        raise RuntimeError(f"Thermal camera disconnected: {thermal.get('error_text', '')}")
    return np.array(thermal["values"], dtype=np.float32)


def read_thermal_summary() -> dict[str, Any]:
    """Read Senxor IR frame and summarise deposit-surface ROI."""
    try:
        matrix = bridge_thermal_matrix()
    except Exception as error:
        print(f"  WARNING: Thermal read failed ({error}) — retrying in 3s...")
        time.sleep(3)
        matrix = bridge_thermal_matrix()
    return summarize_deposit_roi(matrix)


def read_ir_temperature() -> float:
    """Primary IR feedback T_k = roi_mean (deg C)."""
    summary = read_thermal_summary()
    return float(summary["roi_mean"])


def send_gcode_commands(commands: list[str]) -> None:
    if not commands:
        return
    body_key = "command" if len(commands) == 1 else "commands"
    body_value: str | list[str] = commands[0] if len(commands) == 1 else commands
    response = requests.post(
        GCODE_URL,
        json={body_key: body_value},
        timeout=5,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("ok"):
        raise RuntimeError(payload.get("error", "unknown gcode error"))


def build_control_gcode(
    speed_mms: float,
    nozzle_temp_c: float,
    *,
    flow_percent: float = BASE_FLOW_PERCENT,
    base_speed_mms: float = BASE_SPEED_MMS,
) -> list[str]:
    """Build M220/M221/M106/M104 command list (fan locked at 100%, no dwell)."""
    speed_percent = int(round((speed_mms / base_speed_mms) * 100))
    speed_percent = max(10, min(200, speed_percent))
    flow_percent_int = int(round(np.clip(flow_percent, 50, 150)))
    nozzle_temp_int = int(
        round(np.clip(nozzle_temp_c, NOZZLE_TEMP_MIN_C, NOZZLE_TEMP_MAX_C))
    )

    return [
        f"M220 S{speed_percent}",
        f"M221 S{flow_percent_int}",
        f"M106 S{FAN_PWM_FULL}",
        f"M104 S{nozzle_temp_int}",
    ]


def apply_controls(
    speed_mms: float,
    nozzle_temp_c: float,
    *,
    flow_percent: float = BASE_FLOW_PERCENT,
    base_speed_mms: float = BASE_SPEED_MMS,
) -> list[str]:
    """Apply speed and nozzle setpoint via dashboard bridge G-code API."""
    commands = build_control_gcode(
        speed_mms,
        nozzle_temp_c,
        flow_percent=flow_percent,
        base_speed_mms=base_speed_mms,
    )
    send_gcode_commands(commands)
    return commands


def apply_control_vector(
    u: Sequence[float],
    *,
    flow_percent: float = BASE_FLOW_PERCENT,
    base_speed_mms: float = BASE_SPEED_MMS,
) -> list[str]:
    """Apply control vector [v_mm_s, T_nozzle_C]."""
    v, T_nozzle = map(float, u)
    return apply_controls(
        v,
        T_nozzle,
        flow_percent=flow_percent,
        base_speed_mms=base_speed_mms,
    )


def publish_thermal_control_status(
    *,
    control_mode: str,
    speed_mms: float,
    nozzle_temp_c: float,
    print_number: int,
    layer: int | None = None,
    T_k: float | None = None,
    D_k: float | None = None,
    mpc_cost: float | None = None,
    mpc_success: bool | None = None,
    active: bool = True,
) -> None:
    """Publish thermal control status to dashboard bridge."""
    try:
        payload: dict[str, Any] = {
            "active": active,
            "control_mode": control_mode,
            "speed_mms": speed_mms,
            "nozzle_temp_c": nozzle_temp_c,
            "fan_normalized": 1.0,
            "dwell_s": 0.0,
            "base_speed_mms": BASE_SPEED_MMS,
            "print_number": print_number,
            "layer": layer,
        }
        if T_k is not None:
            payload["T_k"] = T_k
        if D_k is not None:
            payload["D_k"] = D_k
        if mpc_cost is not None:
            payload["mpc_cost"] = mpc_cost
        if mpc_success is not None:
            payload["mpc_success"] = mpc_success
        response = requests.post(THERMAL_CONTROL_STATUS_URL, json=payload, timeout=2)
        response.raise_for_status()
    except Exception:
        pass


def clear_thermal_control_status() -> None:
    try:
        response = requests.post(
            THERMAL_CONTROL_STATUS_URL,
            json={"active": False},
            timeout=2,
        )
        response.raise_for_status()
    except Exception:
        pass
