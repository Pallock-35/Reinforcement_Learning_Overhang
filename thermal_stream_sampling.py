"""1 Hz IR stream sampling between layerwise MPC decisions (outline §3.2, §6.6)."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from thermal_dose import ThermalDoseTracker

STREAM_LOG_FILENAME = "stream.csv"
STREAM_INTERVAL_S = float(os.environ.get("PRUSASAM_STREAM_INTERVAL_S", "1.0"))

STREAM_LOG_COLUMNS = [
    "print_number",
    "print_angle_deg",
    "control_mode",
    "last_decision_layer",
    "timestamp",
    "sample_dt_s",
    "elapsed_since_decision_s",
    "T_k",
    "D_k",
    "dT_dt",
    "roi_max",
    "roi_mean",
    "roi_std",
    "roi_min",
    "frame_max",
    "frame_mean",
    "print_speed_mms",
    "nozzle_setpoint_c",
    "fan_percent",
    "dwell_s",
    "hotend_actual",
    "hotend_target",
    "bed_actual",
    "bed_target",
]


@dataclass
class StreamState:
    """Latest IR-derived thermal state from the 1 Hz stream."""

    T_k: float = float("nan")
    D_k: float = 0.0
    dT_dt: float = 0.0
    thermal: dict[str, Any] = field(default_factory=dict)
    sample_monotonic: float = 0.0
    samples_since_decision: int = 0


def create_stream_log_file(
    run_dir_path: str,
) -> tuple[Any, Any, str]:
    """Open stream.csv alongside the layer decision log."""
    from pathlib import Path

    path = str(Path(run_dir_path) / STREAM_LOG_FILENAME)
    handle = open(path, "w", newline="", encoding="utf-8")
    import csv

    writer = csv.writer(handle)
    writer.writerow(STREAM_LOG_COLUMNS)
    handle.flush()
    return handle, writer, path


def sample_stream_tick(
    dose_tracker: ThermalDoseTracker,
    dt_s: float,
    read_thermal_summary: Callable[[], dict[str, Any]],
) -> StreamState:
    """Read T_IR,k, integrate D_k over dt_s, and return stream state."""
    thermal = read_thermal_summary()
    T_k = float(thermal["roi_mean"])
    D_k, dT_dt = dose_tracker.update(T_k, dt_s)
    return StreamState(
        T_k=T_k,
        D_k=D_k,
        dT_dt=dT_dt,
        thermal=thermal,
        sample_monotonic=time.monotonic(),
    )


def write_stream_row(
    writer: Any,
    *,
    print_number: int,
    print_angle_deg: float,
    control_mode: str,
    last_decision_layer: int,
    sample_dt_s: float,
    elapsed_since_decision_s: float,
    stream_state: StreamState,
    print_speed_mms: float,
    nozzle_setpoint_c: float,
    temps: dict[str, Any],
    fan_percent: float = 100.0,
    dwell_s: float = 0.0,
) -> None:
    thermal = stream_state.thermal
    writer.writerow([
        print_number,
        round(print_angle_deg, 2),
        control_mode,
        last_decision_layer,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        round(sample_dt_s, 3),
        round(elapsed_since_decision_s, 3),
        round(stream_state.T_k, 2),
        round(stream_state.D_k, 2),
        round(stream_state.dT_dt, 4),
        round(float(thermal.get("roi_max", float("nan"))), 2),
        round(float(thermal.get("roi_mean", float("nan"))), 2),
        round(float(thermal.get("roi_std", float("nan"))), 2),
        round(float(thermal.get("roi_min", float("nan"))), 2),
        round(float(thermal.get("frame_max", float("nan"))), 2),
        round(float(thermal.get("frame_mean", float("nan"))), 2),
        round(print_speed_mms, 2),
        round(nozzle_setpoint_c, 2),
        round(fan_percent, 2),
        round(dwell_s, 3),
        temps.get("hotend_actual"),
        temps.get("hotend_target"),
        temps.get("bed_actual"),
        temps.get("bed_target"),
    ])


def stream_until_layer(
    watcher: Any,
    *,
    dose_tracker: ThermalDoseTracker,
    stream_writer: Any,
    read_thermal_summary: Callable[[], dict[str, Any]],
    read_printer_temps: Callable[[], dict[str, Any]],
    print_number: int,
    print_angle_deg: float,
    control_mode: str,
    last_decision_layer: int,
    decision_monotonic: float,
    print_speed_mms: float,
    nozzle_setpoint_c: float,
    interval_s: float = STREAM_INTERVAL_S,
    stream_handle: Any | None = None,
    stop_event: Any | None = None,
    fan_percent: float = 100.0,
    dwell_s: float = 0.0,
) -> tuple[int | None, StreamState]:
    """
    Sample IR at ~1 Hz until the next layer completes or the print ends.

    Returns (completed_layer_number | None if print finished, latest stream state).
    """
    last_tick = time.monotonic()
    latest = StreamState()
    samples = 0

    while True:
        if stop_event is not None and stop_event.is_set():
            return None, latest

        layer = watcher.try_pop_layer(timeout=0.2)
        if layer is not None:
            return layer, latest
        if watcher.is_print_finished():
            return None, latest

        now = time.monotonic()
        if now - last_tick < interval_s:
            continue

        dt_s = now - last_tick
        last_tick = now
        latest = sample_stream_tick(dose_tracker, dt_s, read_thermal_summary)
        samples += 1
        latest.samples_since_decision = samples

        try:
            temps = read_printer_temps()
        except Exception:
            temps = {}

        write_stream_row(
            stream_writer,
            print_number=print_number,
            print_angle_deg=print_angle_deg,
            control_mode=control_mode,
            last_decision_layer=last_decision_layer,
            sample_dt_s=dt_s,
            elapsed_since_decision_s=now - decision_monotonic,
            stream_state=latest,
            print_speed_mms=print_speed_mms,
            nozzle_setpoint_c=nozzle_setpoint_c,
            fan_percent=fan_percent,
            dwell_s=dwell_s,
            temps=temps,
        )
        if hasattr(stream_writer, "flush"):
            stream_writer.flush()
        if stream_handle is not None:
            stream_handle.flush()
