#!/usr/bin/env python3
"""
rl_layer_logger.py
==================
Standalone online Q-learning for tower prints with automatic layer detection.

Combines:
  - Layer-triggered CSV logging, webcam photos, and RL control
  - Q-table control of print speed (M220), flow (M221), and fan (M106/M107)
  - IR camera deposit-temperature feedback via Senxor ROI (roi_max)
  - Per-layer webcam snapshots + CNN defect detection (sagging)
  - Optional CNN inference via models/layer_defect_cnn.pt (train with train_layer_defect_cnn.py)

Depends only on dashboard_bridge.py running locally. Thermal reads, printer
temperatures, and G-code overrides all go through the bridge HTTP API.
Layer changes are detected by polling OctoPrint's DisplayLayerProgress plugin
(the bridge does not expose layer events). Webcam stills are fetched directly
from OctoPrint's /webcam/?action=snapshot endpoint.

RL decision loop (each completed layer):
  1. Capture webcam photo and classify normal/defective (CNN or manual labels).
  2. Read IR deposit temperature from the Senxor ROI below the hotend.
  3. Build a discrete state from temperature, speed/flow/fan indices, and visual class.
  4. Pick one of nine actions: nudge speed, flow, or fan down/hold/up.
  5. Reward combines IR temperature band with CNN visual penalty/bonus (RL mode only).

Research focus: maximum overhang angle (from vertical) a closed-loop system can print.
Use --print-angle to record the tower angle for each run. Use --open-loop for fixed
base settings (170 mm/s, 100% flow, 100% fan) with no CNN and no RL control.

Before running:
  1. server.py on the Pi (thermal stream to the bridge)
  2. dashboard_bridge.py on the laptop
  3. DisplayLayerProgress plugin on OctoPrint (recommended)
  4. PRUSASAM_OCTOPRINT_API_KEY env var or api_key.py for layer polling

Usage:
  python rl_layer_logger.py
  python rl_layer_logger.py --debug-layers
  python rl_layer_logger.py --layer-labels "1-30:normal,31-50:defective"
  python rl_layer_logger.py --layer-labels-file layer_labels.json
  python rl_layer_logger.py --apply-labels-to print_runs/print_001/log.csv --layer-labels "1-30:normal,31-50:defective"
  python train_layer_defect_cnn.py --photo-root print_runs --layer-labels-file layer_labels.json
  python rl_layer_logger.py --apply-cnn-to print_runs/print_001/log.csv
  python rl_layer_logger.py --print-angle 30 --open-loop
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import queue
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import requests


# ══════════════════════════════════════════════════════════════════
#  SECTION 1 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════

BRIDGE_URL = os.environ.get("PRUSASAM_BRIDGE_HOST", "http://127.0.0.1:8765")
BRIDGE_URL = (
    BRIDGE_URL
    if BRIDGE_URL.startswith("http")
    else f"http://{BRIDGE_URL}:8765"
).rstrip("/")
THERMAL_URL = f"{BRIDGE_URL}/api/thermal"
DASHBOARD_URL = f"{BRIDGE_URL}/api/dashboard"
GCODE_URL = f"{BRIDGE_URL}/api/gcode"

DEFAULT_OCTOPRINT_URL = "http://cimpi-prusamk4s.local"
DISPLAY_LAYER_PROGRESS_PATH = "/plugin/DisplayLayerProgress/values"

from thermal_roi import THERMAL_COLS, THERMAL_ROWS, summarize_deposit_roi

BASE_SPEED_MMS = 170.0
BASE_FLOW_PERCENT = 100.0
BASE_FAN_PERCENT = 100.0
SPEED_SETTLE_TIME = 3.0
LAYER_HEIGHT = 0.2
LAYER_WAIT_TIMEOUT = 600.0
LAYER_POLL_INTERVAL = 1.0

BAND_LO = 60.0
BAND_HI = 90.0
SPEEDS = np.array([65.0, 80.0, 100.0, 120.0, 140.0, 160.0], dtype=float)
FLOWS = np.array([85.0, 90.0, 95.0, 100.0, 105.0, 110.0], dtype=float)
FANS = np.array([0.0, 25.0, 50.0, 75.0, 100.0], dtype=float)

CNN_DEFECT_PENALTY = 2.0
CNN_NORMAL_BONUS = 0.25

ALPHA = 0.05
GAMMA = 0.95
EPSILON_START = 0.30
EPSILON_MIN = 0.05
N_PRINTS = 20

N_VISUAL_BINS = 3
N_STATES = 5 * len(SPEEDS) * len(FLOWS) * len(FANS) * N_VISUAL_BINS
N_ACTIONS = 9

ACTION_SPECS: list[tuple[str, int, str]] = [
    ("speed", -1, "speed down"),
    ("speed", 0, "speed hold "),
    ("speed", 1, "speed up  "),
    ("flow", -1, "flow down "),
    ("flow", 0, "flow hold "),
    ("flow", 1, "flow up   "),
    ("fan", -1, "fan down  "),
    ("fan", 0, "fan hold  "),
    ("fan", 1, "fan up    "),
]

Q_TABLE_DIR = "q_tables"
from print_run_naming import (
    PRINT_RUNS_DIR,
    normalize_print_run_key,
    parse_print_number_from_run_key,
    print_run_dir,
    print_run_folder_name,
    print_run_photos_dir,
)
PRINT_PHOTOS_SUBDIR = "photos"
PRINT_LOG_FILENAME = "log.csv"
LAYER_PHOTO_ENABLED = True
LAYER_PHOTO_SETTLE_TIME = 1.0
LAYER_DEFECT_MODEL_PATH = Path(
    os.environ.get("PRUSASAM_LAYER_DEFECT_MODEL", "models/layer_defect_cnn.pt")
)
CNN_MIN_CONFIDENCE = float(os.environ.get("PRUSASAM_CNN_MIN_CONFIDENCE", "0.5"))
CNN_ENABLED = True

DISPLAY_LAYER_RE = re.compile(
    r"(?:L\s*=\s*|Layer[:\s]+)(\d+)\s*/\s*(\d+)",
    re.IGNORECASE,
)
LAYER_LABEL_PAIR_RE = re.compile(
    r"^\s*(\d+)(?:\s*-\s*(\d+))?\s*:\s*(\w+)\s*$",
    re.IGNORECASE,
)
DEBUG_LAYERS = True


def _debug(message: str) -> None:
    if DEBUG_LAYERS:
        print(f"  [layer-detect] {message}")


def load_octoprint_api_key() -> str:
    key = os.environ.get("PRUSASAM_OCTOPRINT_API_KEY", "").strip()
    if key:
        return key
    try:
        from api_key import API_key

        return API_key
    except ImportError:
        return ""


def fetch_octoprint_json(url: str, api_key: str) -> dict[str, Any]:
    request = Request(url, headers={"X-Api-Key": api_key})
    with urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


# ══════════════════════════════════════════════════════════════════
#  SECTION 2 — DASHBOARD BRIDGE CLIENT
# ══════════════════════════════════════════════════════════════════

def bridge_health() -> bool:
    response = requests.get(f"{BRIDGE_URL}/api/health", timeout=5)
    return response.status_code == 200


def bridge_dashboard() -> dict[str, Any]:
    response = requests.get(DASHBOARD_URL, timeout=10)
    response.raise_for_status()
    return response.json()


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


def read_thermal() -> dict[str, float]:
    """Read Senxor IR frame and summarise deposit-surface temperature in the ROI."""
    try:
        matrix = bridge_thermal_matrix()
    except Exception as error:
        print(f"  WARNING: Thermal read failed ({error}) — retrying in 3s...")
        time.sleep(3)
        matrix = bridge_thermal_matrix()

    return summarize_deposit_roi(matrix)


def read_printer_temps() -> dict[str, float | None]:
    dashboard = bridge_dashboard()
    temps = dashboard.get("temperatures", {})
    hotend = temps.get("tool0", {})
    bed = temps.get("bed", {})
    return {
        "hotend_actual": hotend.get("actual"),
        "hotend_target": hotend.get("target"),
        "bed_actual": bed.get("actual"),
        "bed_target": bed.get("target"),
        "is_printing": bool(dashboard.get("is_printing", False)),
        "octoprint_url": dashboard.get("octoprint_url", DEFAULT_OCTOPRINT_URL),
    }


def send_gcode_commands(commands: list[str]) -> None:
    if not commands:
        return
    body_key = "command" if len(commands) == 1 else "commands"
    body_value: str | list[str] = commands[0] if len(commands) == 1 else commands
    try:
        response = requests.post(
            GCODE_URL,
            json={body_key: body_value},
            timeout=5,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok"):
            raise RuntimeError(payload.get("error", "unknown gcode error"))
    except Exception as error:
        print(f"  WARNING: G-code command failed — {error}")


def fan_percent_to_m106(fan_percent: float) -> int:
    return int(round(np.clip(fan_percent, 0.0, 100.0) / 100.0 * 255))


def send_print_controls(
    speed_mms: float,
    flow_percent: float,
    fan_percent: float,
) -> None:
    speed_percent = int(round((speed_mms / BASE_SPEED_MMS) * 100))
    speed_percent = max(10, min(200, speed_percent))
    flow_percent_int = int(round(np.clip(flow_percent, 50, 150)))
    commands = [
        f"M220 S{speed_percent}",
        f"M221 S{flow_percent_int}",
    ]
    fan_pwm = fan_percent_to_m106(fan_percent)
    if fan_pwm <= 0:
        commands.append("M107")
    else:
        commands.append(f"M106 S{fan_pwm}")
    send_gcode_commands(commands)


def send_speed_command(speed_mms: float) -> None:
    send_print_controls(speed_mms, BASE_FLOW_PERCENT, BASE_FAN_PERCENT)


RL_STATUS_URL = f"{BRIDGE_URL}/api/rl-status"


def publish_rl_speed_status(
    speed_mms: float,
    print_number: int,
    layer: int | None = None,
    *,
    active: bool = True,
    flow_percent: float | None = None,
    fan_percent: float | None = None,
    visual_defect: str | None = None,
) -> None:
    """Publish the RL agent's current controls to dashboard_bridge for the dashboard UI."""
    try:
        payload: dict[str, Any] = {
            "active": active,
            "speed_mms": speed_mms,
            "base_speed_mms": BASE_SPEED_MMS,
            "print_number": print_number,
            "layer": layer,
        }
        if flow_percent is not None:
            payload["flow_percent"] = flow_percent
        if fan_percent is not None:
            payload["fan_percent"] = fan_percent
        if visual_defect:
            payload["visual_defect"] = visual_defect
        response = requests.post(RL_STATUS_URL, json=payload, timeout=2)
        response.raise_for_status()
    except Exception:
        pass


def clear_rl_speed_status() -> None:
    try:
        response = requests.post(RL_STATUS_URL, json={"active": False}, timeout=2)
        response.raise_for_status()
    except Exception:
        pass


class RlSpeedStatusPublisher:
    """Keep dashboard_bridge updated while an RL print episode is running."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._speed_mms = BASE_SPEED_MMS
        self._flow_percent = BASE_FLOW_PERCENT
        self._fan_percent = BASE_FAN_PERCENT
        self._visual_defect: str | None = None
        self._print_number = 0
        self._layer: int | None = None

    def start(
        self,
        print_number: int,
        initial_speed_mms: float,
        *,
        initial_flow_percent: float = BASE_FLOW_PERCENT,
        initial_fan_percent: float = BASE_FAN_PERCENT,
    ) -> None:
        self._print_number = print_number
        self._speed_mms = initial_speed_mms
        self._flow_percent = initial_flow_percent
        self._fan_percent = initial_fan_percent
        self._visual_defect = None
        self._layer = None
        self._stop.clear()
        self.publish()
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._thread.start()

    def update(
        self,
        speed_mms: float,
        layer: int | None = None,
        *,
        flow_percent: float | None = None,
        fan_percent: float | None = None,
        visual_defect: str | None = None,
    ) -> None:
        self._speed_mms = speed_mms
        if flow_percent is not None:
            self._flow_percent = flow_percent
        if fan_percent is not None:
            self._fan_percent = fan_percent
        if visual_defect is not None:
            self._visual_defect = visual_defect or None
        if layer is not None:
            self._layer = layer
        self.publish()

    def publish(self) -> None:
        publish_rl_speed_status(
            self._speed_mms,
            self._print_number,
            self._layer,
            active=True,
            flow_percent=self._flow_percent,
            fan_percent=self._fan_percent,
            visual_defect=self._visual_defect,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        clear_rl_speed_status()

    def _heartbeat(self) -> None:
        while not self._stop.wait(5.0):
            self.publish()


def build_webcam_snapshot_url(octoprint_url: str) -> str:
    return f"{octoprint_url.rstrip('/')}/webcam/?action=snapshot"


PRINT_RUN_META_FILENAME = "run_meta.json"


def base_print_settings() -> tuple[float, float, float]:
    """Fixed open-loop controls: 170 mm/s, 100% flow, 100% fan."""
    return BASE_SPEED_MMS, BASE_FLOW_PERCENT, BASE_FAN_PERCENT


def base_control_indices() -> tuple[int, int, int]:
    """Discrete RL indices closest to the base open-loop settings."""
    speed_index = int(np.argmin(np.abs(SPEEDS - BASE_SPEED_MMS)))
    flow_index = int(np.argmin(np.abs(FLOWS - BASE_FLOW_PERCENT)))
    fan_index = int(np.argmin(np.abs(FANS - BASE_FAN_PERCENT)))
    return speed_index, flow_index, fan_index


def write_print_run_metadata(
    print_number: int,
    *,
    print_angle_deg: float,
    control_mode: str,
    session_filename: str = "",
    runs_root: Path | str = PRINT_RUNS_DIR,
) -> Path:
    run_dir = print_run_dir(
        print_number,
        runs_root,
        control_mode=control_mode,
        print_angle_deg=print_angle_deg,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / PRINT_RUN_META_FILENAME
    run_folder = print_run_folder_name(
        print_number,
        control_mode=control_mode,
        print_angle_deg=print_angle_deg,
    )
    payload = {
        "print_number": print_number,
        "print_angle_deg": print_angle_deg,
        "control_mode": control_mode,
        "run_folder": run_folder,
        "base_speed_mms": BASE_SPEED_MMS,
        "base_flow_percent": BASE_FLOW_PERCENT,
        "base_fan_percent": BASE_FAN_PERCENT,
        "session_file": os.path.basename(session_filename) if session_filename else "",
        "recorded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return meta_path


def layer_photo_path_for(
    print_number: int,
    layer: int,
    *,
    control_mode: str | None = None,
    print_angle_deg: float | None = None,
    runs_root: Path | str = PRINT_RUNS_DIR,
) -> Path:
    return print_run_photos_dir(
        print_number,
        runs_root,
        control_mode=control_mode,
        print_angle_deg=print_angle_deg,
    ) / f"layer_{layer:04d}.jpg"


@dataclass(frozen=True)
class VisualDefectAssessment:
    """Visual quality assessment for a completed layer."""

    label: str | None = None
    confidence: float | None = None
    source: str | None = None


def normalize_visual_label(raw: str) -> str:
    token = raw.strip().lower()
    if token in {"normal", "ok", "good"}:
        return "normal"
    if token in {"defective", "defect", "sagging", "bad"}:
        return "defective"
    raise ValueError(f"Unknown layer label '{raw}'. Use normal or defective.")


@dataclass(frozen=True)
class LayerLabelRange:
    start: int
    end: int
    label: str


class LayerLabelMap:
    """Map completed layer numbers to manual normal/defective labels."""

    def __init__(self, ranges: list[LayerLabelRange] | None = None) -> None:
        self._ranges = sorted(ranges or [], key=lambda item: (item.start, item.end))

    def __bool__(self) -> bool:
        return bool(self._ranges)

    def lookup(self, layer: int) -> VisualDefectAssessment | None:
        for item in self._ranges:
            if item.start <= layer <= item.end:
                return VisualDefectAssessment(
                    label=item.label,
                    confidence=1.0,
                    source="manual",
                )
        return None

    def describe(self) -> str:
        if not self._ranges:
            return "(none)"
        parts = []
        for item in self._ranges:
            if item.start == item.end:
                parts.append(f"{item.start}:{item.label}")
            else:
                parts.append(f"{item.start}-{item.end}:{item.label}")
        return ", ".join(parts)

    @classmethod
    def parse(cls, spec: str) -> LayerLabelMap:
        spec = spec.strip()
        if not spec:
            return cls()

        ranges: list[LayerLabelRange] = []
        for chunk in spec.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            match = LAYER_LABEL_PAIR_RE.match(chunk)
            if not match:
                raise ValueError(
                    f"Invalid layer label '{chunk}'. "
                    "Use RANGE:label entries like 1-30:normal or 42:defective."
                )
            start = int(match.group(1))
            end = int(match.group(2) or start)
            if end < start:
                raise ValueError(f"Invalid layer range in '{chunk}': end < start.")
            label = normalize_visual_label(match.group(3))
            ranges.append(LayerLabelRange(start=start, end=end, label=label))
        return cls(ranges)


@dataclass
class LayerLabelSession:
    """Per-print manual labels loaded from CLI and/or a JSON file."""

    default: LayerLabelMap | None = None
    per_run: dict[str, LayerLabelMap] | None = None

    def for_print(
        self,
        print_number: int,
        *,
        control_mode: str | None = None,
        print_angle_deg: float | None = None,
    ) -> LayerLabelMap:
        if control_mode is not None and print_angle_deg is not None:
            folder = print_run_folder_name(
                print_number,
                control_mode=control_mode,
                print_angle_deg=print_angle_deg,
            )
            if self.per_run and folder in self.per_run:
                return self.per_run[folder]
        folder = print_run_folder_name(print_number)
        if self.per_run:
            if folder in self.per_run:
                return self.per_run[folder]
            spaced = f"print {print_number}"
            if spaced in self.per_run:
                return self.per_run[spaced]
            for key, label_map in self.per_run.items():
                if parse_print_number_from_run_key(key) == print_number:
                    return label_map
        return self.default or LayerLabelMap()

    def for_run_folder(self, folder_name: str) -> LayerLabelMap:
        if not self.per_run:
            return self.default or LayerLabelMap()
        if folder_name in self.per_run:
            return self.per_run[folder_name]
        canonical = normalize_print_run_key(folder_name)
        if canonical in self.per_run:
            return self.per_run[canonical]
        return self.default or LayerLabelMap()

    @classmethod
    def from_specs(
        cls,
        *,
        default_spec: str | None = None,
        labels_file: str | Path | None = None,
    ) -> LayerLabelSession:
        session = cls()
        if labels_file:
            session = cls.load_file(labels_file)
        if default_spec:
            session.default = LayerLabelMap.parse(default_spec)
        return session

    @classmethod
    def load_file(cls, path: str | Path) -> LayerLabelSession:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Layer label file must contain a JSON object.")

        default = None
        if "default" in payload and payload["default"]:
            default = LayerLabelMap.parse(_coerce_label_spec(payload["default"]))

        per_run: dict[str, LayerLabelMap] = {}
        prints_payload = payload.get("prints", {})
        if prints_payload:
            if not isinstance(prints_payload, dict):
                raise ValueError("'prints' must be an object keyed by print folder name.")
            for print_key, print_spec in prints_payload.items():
                key = normalize_print_run_key(str(print_key))
                per_run[key] = LayerLabelMap.parse(_coerce_label_spec(print_spec))

        reserved = {"default", "prints"}
        for key, value in payload.items():
            if key in reserved:
                continue
            if isinstance(value, (str, list)) and value:
                run_key = normalize_print_run_key(key)
                per_run[run_key] = LayerLabelMap.parse(_coerce_label_spec(value))

        return cls(default=default, per_run=per_run or None)


def _coerce_label_spec(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return ",".join(str(item).strip() for item in value if str(item).strip())
    raise ValueError("Layer label specs must be a string or list of strings.")


@dataclass
class CnnSettings:
    enabled: bool = True
    model_path: Path = LAYER_DEFECT_MODEL_PATH
    min_confidence: float = CNN_MIN_CONFIDENCE
    device: str | None = None


_CNN_SETTINGS = CnnSettings()


def configure_cnn(
    *,
    enabled: bool | None = None,
    model_path: str | Path | None = None,
    min_confidence: float | None = None,
    device: str | None = None,
) -> None:
    global _CNN_SETTINGS
    current = _CNN_SETTINGS
    _CNN_SETTINGS = CnnSettings(
        enabled=current.enabled if enabled is None else enabled,
        model_path=Path(model_path) if model_path is not None else current.model_path,
        min_confidence=current.min_confidence if min_confidence is None else min_confidence,
        device=current.device if device is None else device,
    )


def assess_layer_photo(photo_path: str | Path) -> VisualDefectAssessment:
    """Classify a layer photo with the trained CNN when a model is available."""
    if not _CNN_SETTINGS.enabled or not _CNN_SETTINGS.model_path.exists():
        return VisualDefectAssessment()
    try:
        from layer_defect_cnn import predict_layer_photo

        prediction = predict_layer_photo(
            photo_path,
            model_path=_CNN_SETTINGS.model_path,
            min_confidence=_CNN_SETTINGS.min_confidence,
            device=_CNN_SETTINGS.device,
        )
        if prediction is None:
            return VisualDefectAssessment()
        return VisualDefectAssessment(
            label=prediction.label,
            confidence=prediction.confidence,
            source="cnn",
        )
    except Exception as error:
        print(f"  WARNING: CNN prediction failed ({error})")
        return VisualDefectAssessment()


def assess_layer_visual(
    layer: int,
    photo_path: str | Path | None,
    *,
    manual_labels: LayerLabelMap | None = None,
) -> VisualDefectAssessment:
    if manual_labels:
        manual = manual_labels.lookup(layer)
        if manual is not None:
            return manual
    if photo_path:
        cnn = assess_layer_photo(photo_path)
        if cnn.label is not None:
            return VisualDefectAssessment(
                label=cnn.label,
                confidence=cnn.confidence,
                source="cnn",
            )
    return VisualDefectAssessment()


def apply_layer_labels_to_csv(
    csv_path: str | Path,
    label_map: LayerLabelMap,
) -> int:
    """Fill visual_defect columns in an existing log using manual layer ranges."""
    path = Path(csv_path)
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header found in {path}")
        fieldnames = list(reader.fieldnames)
        for column in ("visual_defect", "visual_confidence", "visual_defect_source"):
            if column not in fieldnames:
                fieldnames.append(column)
        for row in reader:
            rows.append(row)

    updated = 0
    for row in rows:
        layer_text = row.get("layer", "").strip()
        if not layer_text.isdigit():
            continue
        assessment = label_map.lookup(int(layer_text))
        if assessment is None:
            continue
        row["visual_defect"] = assessment.label or ""
        row["visual_confidence"] = (
            "" if assessment.confidence is None else f"{assessment.confidence:.4f}"
        )
        row["visual_defect_source"] = assessment.source or ""
        updated += 1

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return updated


def capture_layer_photo(
    octoprint_url: str,
    print_number: int,
    layer: int,
    *,
    control_mode: str | None = None,
    print_angle_deg: float | None = None,
    runs_root: Path | str = PRINT_RUNS_DIR,
    settle_time: float = LAYER_PHOTO_SETTLE_TIME,
) -> str | None:
    """Save a JPEG snapshot from the OctoPrint webcam for the completed layer."""
    if settle_time > 0:
        time.sleep(settle_time)

    snapshot_url = build_webcam_snapshot_url(octoprint_url)
    try:
        response = requests.get(
            snapshot_url,
            timeout=10,
            headers={"User-Agent": "PrusaSAM/1.0"},
        )
        response.raise_for_status()
        content = response.content
        if not content.startswith(b"\xff\xd8"):
            raise RuntimeError("snapshot response is not a JPEG")
    except Exception as error:
        print(f"  WARNING: Layer photo capture failed ({error})")
        return None

    path = layer_photo_path_for(
        print_number,
        layer,
        control_mode=control_mode,
        print_angle_deg=print_angle_deg,
        runs_root=runs_root,
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    except Exception as error:
        print(f"  WARNING: Layer photo save failed ({error})")
        return None

    return str(path)


def inspect_thermal_frame() -> None:
    matrix = bridge_thermal_matrix()
    print(f"\nThermal frame  ({matrix.shape[0]} rows x {matrix.shape[1]} cols)")
    print(
        f"  Full frame:  min={matrix.min():.1f}  "
        f"max={matrix.max():.1f}  mean={matrix.mean():.1f}  deg C"
    )
    summary = summarize_deposit_roi(matrix)
    roi_start = int(summary["roi_row_start"])
    roi_end = int(summary["roi_row_end"])
    col_start = int(summary["roi_col_start"])
    col_end = int(summary["roi_col_end"])
    print(
        f"\n  Deposit ROI: rows {roi_start}–{roi_end - 1}, cols {col_start}–{col_end - 1} "
        f"(filament pixels={summary.get('filament_pixel_count', '?')}, "
        f"tri={summary.get('hotend_triangle_h', '?')}x{summary.get('hotend_triangle_w', '?')}, "
        f"trim top {summary.get('roi_trim_top_frac', 0.1) * 100:.0f}%)"
    )
    print("\nRow-by-row max — hotend rows should stay above the ROI band:")
    for row in range(min(40, THERMAL_ROWS)):
        row_max = float(matrix[row, :].max())
        spread = max(1.0, matrix.max() - matrix.min())
        bar = "█" * int((row_max - matrix.min()) / spread * 30)
        if row < roi_start:
            zone = "<- SKIP  (hotend)"
        elif row < roi_end:
            zone = "<- ROI   (layer surface)"
        else:
            zone = ""
        print(f"  {row:>4}  {row_max:>6.1f}C  {bar:<32}  {zone}")


def test_connections() -> bool:
    print("\n" + "=" * 55)
    print("  CONNECTION TEST")
    print("=" * 55)
    all_ok = True

    print("\n  1. dashboard_bridge.py...")
    try:
        if bridge_health():
            print(f"     OK  — {BRIDGE_URL}")
        else:
            print("     Bridge health check failed.")
            all_ok = False
    except Exception as error:
        print(f"     FAILED — {error}")
        all_ok = False

    print("\n  2. Thermal camera...")
    try:
        reading = read_thermal()
        print(
            f"     OK  — roi_max={reading['roi_max']:.1f}C  "
            f"roi_mean={reading['roi_mean']:.1f}C"
        )
    except Exception as error:
        print(f"     FAILED — {error}")
        all_ok = False

    print("\n  3. OctoPrint via bridge dashboard...")
    try:
        status = read_printer_temps()
        print(f"     OK  — {status['octoprint_url']}")
        print(
            f"       hotend={status['hotend_actual']}C  "
            f"bed={status['bed_actual']}C  "
            f"printing={status['is_printing']}"
        )
    except Exception as error:
        print(f"     FAILED — {error}")
        all_ok = False

    api_key = load_octoprint_api_key()
    print("\n  4. DisplayLayerProgress layer API...")
    if not api_key:
        print("     SKIPPED — set PRUSASAM_OCTOPRINT_API_KEY or api_key.py")
    else:
        try:
            octoprint_url = read_printer_temps()["octoprint_url"] or DEFAULT_OCTOPRINT_URL
            payload = fetch_display_layer_progress(octoprint_url, api_key)
            if payload is None:
                print("     WARNING — plugin not found; layer timing may fail")
            else:
                layer, height = parse_display_layer_progress(payload)
                print(f"     OK  — layer={layer}  height={height}")
        except Exception as error:
            print(f"     FAILED — {error}")
            all_ok = False

    if LAYER_PHOTO_ENABLED:
        print("\n  5. OctoPrint webcam snapshot...")
        try:
            octoprint_url = read_printer_temps()["octoprint_url"] or DEFAULT_OCTOPRINT_URL
            snapshot_url = build_webcam_snapshot_url(octoprint_url)
            response = requests.get(
                snapshot_url,
                timeout=10,
                headers={"User-Agent": "PrusaSAM/1.0"},
            )
            response.raise_for_status()
            if response.content.startswith(b"\xff\xd8"):
                print(f"     OK  — {len(response.content) // 1024} KB from {snapshot_url}")
            else:
                print("     WARNING — snapshot did not return JPEG data")
        except Exception as error:
            print(f"     WARNING — {error} (layer photos will be skipped if this persists)")

    if CNN_ENABLED and _CNN_SETTINGS.model_path.exists():
        print(f"\n  6. Layer defect CNN ({_CNN_SETTINGS.model_path})...")
        print("     OK  — model file found")
    elif CNN_ENABLED:
        print("\n  6. Layer defect CNN...")
        print(
            f"     SKIPPED — no model at {_CNN_SETTINGS.model_path}; "
            "train with train_layer_defect_cnn.py"
        )

    print()
    print("  All connections OK." if all_ok else "  Fix issues above before printing.")
    print("=" * 55 + "\n")
    return all_ok


# ══════════════════════════════════════════════════════════════════
#  SECTION 3 — LAYER CHANGE DETECTION (OctoPrint poll)
# ══════════════════════════════════════════════════════════════════

def _parse_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
        try:
            return int(float(stripped))
        except ValueError:
            return None
    return None


def _parse_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def fetch_display_layer_progress(base_url: str, api_key: str) -> dict[str, Any] | None:
    url = f"{base_url.rstrip('/')}{DISPLAY_LAYER_PROGRESS_PATH}"
    try:
        return fetch_octoprint_json(url, api_key)
    except HTTPError as error:
        if error.code in {404, 501}:
            return None
        raise


def parse_display_layer_progress(payload: dict[str, Any]) -> tuple[int | None, float | None]:
    layer_info = payload.get("layer")
    height_info = payload.get("height")
    current_layer = None
    current_height = None
    if isinstance(layer_info, dict):
        current_layer = _parse_int(layer_info.get("current"))
    if isinstance(height_info, dict):
        current_height = _parse_float(height_info.get("current"))
    if current_layer is None:
        current_layer = _parse_int(payload.get("currentLayer"))
    if current_height is None:
        current_height = _parse_float(payload.get("currentHeight"))
    return current_layer, current_height


class LayerTracker:
    def __init__(self) -> None:
        self._last_seen_layer = 0
        self._last_emitted_layer = 0
        self._last_height_layer = 0

    def consume_layer_started(self, current_layer: int) -> int | None:
        if current_layer <= self._last_seen_layer:
            return None
        completed = self._last_seen_layer if self._last_seen_layer >= 1 else None
        self._last_seen_layer = current_layer
        return self._emit_if_new(completed)

    def consume_height(self, height_mm: float, layer_height: float) -> int | None:
        if layer_height <= 0 or height_mm <= 0:
            return None
        layers_started = max(1, int(round(height_mm / layer_height)))
        if layers_started <= self._last_height_layer:
            return None
        self._last_height_layer = layers_started
        if layers_started < 2:
            return None
        return self._emit_if_new(layers_started - 1)

    def _emit_if_new(self, layer_number: int | None) -> int | None:
        if layer_number is None or layer_number <= self._last_emitted_layer:
            return None
        self._last_emitted_layer = layer_number
        return layer_number


class LayerChangeWatcher:
    """Poll OctoPrint DisplayLayerProgress and the bridge for layer completion."""

    def __init__(
        self,
        octoprint_url: str,
        api_key: str,
        layer_height: float = LAYER_HEIGHT,
        poll_interval: float = LAYER_POLL_INTERVAL,
    ) -> None:
        self.octoprint_url = octoprint_url.rstrip("/")
        self.api_key = api_key
        self.layer_height = layer_height
        self.poll_interval = poll_interval
        self._tracker = LayerTracker()
        self._layer_queue: queue.Queue[int] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._was_printing = False
        self._print_finished = False
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def is_print_finished(self) -> bool:
        with self._lock:
            return self._print_finished

    def wait_for_print_start(self) -> None:
        print("  Waiting for active print (via bridge dashboard)...")

        while not self._stop.is_set():
            try:
                status = read_printer_temps()

                if status["is_printing"]:

                    # Start every new print with a clean layer tracker.
                    # This prevents stale layer values from the previous
                    # OctoPrint job from blocking the new print.
                    self._tracker = LayerTracker()

                    # Remove any stale queued layer events.
                    while True:
                        try:
                            self._layer_queue.get_nowait()
                        except queue.Empty:
                            break

                    with self._lock:
                        self._was_printing = True
                        self._print_finished = False

                    print(
                        "  Print detected — layer tracker reset; "
                        "monitoring layer changes."
                    )
                    return

            except Exception as error:
                print(
                    f"  WARNING: dashboard poll failed "
                    f"({error})"
                )

            time.sleep(self.poll_interval)

        raise RuntimeError(
            "Layer watcher stopped before print began."
        )

    def wait_for_layer_complete(self, timeout: float = LAYER_WAIT_TIMEOUT) -> int | None:
        deadline = time.monotonic() + timeout
        while not self._stop.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"No layer change within {timeout:.0f}s. "
                    "Install DisplayLayerProgress or check LAYER_HEIGHT."
                )
            layer = self.try_pop_layer(timeout=min(remaining, 1.0))
            if layer is not None:
                return layer
            if self.is_print_finished():
                return None
        return None

    def try_pop_layer(self, timeout: float = 0.0) -> int | None:
        """Return a completed layer number if one is queued, else None."""
        try:
            return self._layer_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _set_print_finished(self) -> None:
        with self._lock:
            self._print_finished = True

    def _emit_layer_complete(self, layer_number: int | None, source: str) -> None:
        if layer_number is None:
            return
        _debug(f"{source}: layer {layer_number} complete")
        self._layer_queue.put(layer_number)

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                status = read_printer_temps()
                is_printing = bool(
                    status["is_printing"]
                )

                if is_printing:
                    with self._lock:
                        self._was_printing = True
                        self._print_finished = False

                elif self._was_printing:
                    self._set_print_finished()

                # IMPORTANT:
                # Read layer progress only while a print is active.
                # DisplayLayerProgress can retain the final layer
                # number from the previous print while OctoPrint is idle.
                if is_printing and self.api_key:
                    payload = fetch_display_layer_progress(
                        self.octoprint_url,
                        self.api_key,
                    )

                    if payload is not None:
                        current_layer, current_height = (
                            parse_display_layer_progress(
                                payload
                            )
                        )

                        if current_layer is not None:
                            completed = (
                                self._tracker
                                .consume_layer_started(
                                    current_layer
                                )
                            )

                            self._emit_layer_complete(
                                completed,
                                "DLP layer",
                            )

                        elif current_height is not None:
                            completed = (
                                self._tracker.consume_height(
                                    current_height,
                                    self.layer_height,
                                )
                            )

                            self._emit_layer_complete(
                                completed,
                                "DLP height",
                            )

            except Exception as error:
                _debug(f"poll error: {error}")

            if self._stop.wait(
                self.poll_interval
            ):
                break


# ══════════════════════════════════════════════════════════════════
#  SECTION 4 — RL SUPPORT
# ══════════════════════════════════════════════════════════════════

def temp_bin(temp: float) -> int:
    if temp < 40:
        return 0
    if temp < 60:
        return 1
    if temp <= 90:
        return 2
    if temp <= 110:
        return 3
    return 4


def visual_bin(visual: VisualDefectAssessment) -> int:
    if visual.label == "normal":
        return 1
    if visual.label == "defective":
        return 2
    return 0


def discretise(
    temp: float,
    speed_index: int,
    flow_index: int,
    fan_index: int,
    visual: VisualDefectAssessment,
) -> int:
    temp_index = temp_bin(temp)
    visual_index = visual_bin(visual)
    return (
        (((temp_index * len(SPEEDS) + speed_index) * len(FLOWS) + flow_index) * len(FANS) + fan_index)
        * N_VISUAL_BINS
        + visual_index
    )


def apply_action(
    action: int,
    speed_index: int,
    flow_index: int,
    fan_index: int,
) -> tuple[int, int, int]:
    parameter, delta, _name = ACTION_SPECS[action]
    if parameter == "speed":
        speed_index = int(np.clip(speed_index + delta, 0, len(SPEEDS) - 1))
    elif parameter == "flow":
        flow_index = int(np.clip(flow_index + delta, 0, len(FLOWS) - 1))
    else:
        fan_index = int(np.clip(fan_index + delta, 0, len(FANS) - 1))
    return speed_index, flow_index, fan_index


def calculate_reward(
    temp: float,
    visual: VisualDefectAssessment | None = None,
) -> float:
    if BAND_LO <= temp <= BAND_HI:
        reward = 1.0
    elif temp < BAND_LO:
        reward = -0.05 * (BAND_LO - temp)
    else:
        reward = -0.05 * (temp - BAND_HI)

    if visual is not None and visual.label == "defective":
        confidence = visual.confidence if visual.confidence is not None else 1.0
        reward -= CNN_DEFECT_PENALTY * confidence
    elif (
        visual is not None
        and visual.label == "normal"
        and BAND_LO <= temp <= BAND_HI
    ):
        reward += CNN_NORMAL_BONUS
    return reward


def get_session_filename() -> str:
    os.makedirs(Q_TABLE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(Q_TABLE_DIR, f"q_table_{timestamp}.npy")


def load_q_table() -> np.ndarray:
    os.makedirs(Q_TABLE_DIR, exist_ok=True)
    all_files = sorted(glob.glob(os.path.join(Q_TABLE_DIR, "q_table_[0-9]*.npy")))
    if not all_files:
        print("  No Q-table found — starting from zero.")
        return np.zeros((N_STATES, N_ACTIONS))
    for candidate in reversed(all_files):
        try:
            q_table = np.load(candidate)
            if q_table.shape != (N_STATES, N_ACTIONS):
                print(
                    f"  WARNING: {candidate} has shape {q_table.shape}, "
                    f"expected ({N_STATES}, {N_ACTIONS}); skipping."
                )
                continue
            print(f"  Loaded: {candidate}")
            return q_table
        except Exception as error:
            print(f"  WARNING: {candidate} corrupted ({error})")
    return np.zeros((N_STATES, N_ACTIONS))


def save_q_table(q_table: np.ndarray, session_filename: str) -> None:
    try:
        np.save(session_filename, q_table)
        np.save(os.path.join(Q_TABLE_DIR, "q_table_latest.npy"), q_table)
    except Exception as error:
        print(f"  WARNING: Q-table save failed — {error}")


def create_log_file(
    print_number: int,
    session_filename: str,
    *,
    print_angle_deg: float,
    control_mode: str,
):
    run_dir = print_run_dir(
        print_number,
        control_mode=control_mode,
        print_angle_deg=print_angle_deg,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    filename = str(run_dir / PRINT_LOG_FILENAME)
    handle = open(filename, "w", newline="", encoding="utf-8")
    writer = csv.writer(handle)
    writer.writerow([
        "print_number",
        "print_angle_deg",
        "control_mode",
        "layer",
        "timestamp",
        "print_speed_mms",
        "flow_percent",
        "fan_percent",
        "deposit_temp_c",
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
        "roi_skip_top",
        "roi_n_rows",
        "current_state",
        "chosen_action",
        "action_name",
        "how",
        "new_roi_max",
        "new_roi_mean",
        "new_frame_max",
        "new_state",
        "reward",
        "in_window",
        "q_value_before",
        "q_value_after",
        "epsilon",
        "session_file",
        "layer_photo",
        "visual_defect",
        "visual_confidence",
        "visual_defect_source",
    ])
    handle.flush()
    print(f"  Logging to: {filename}")
    return handle, writer, filename


# ══════════════════════════════════════════════════════════════════
#  SECTION 5 — ONE PRINT EPISODE
# ══════════════════════════════════════════════════════════════════

def run_one_print(
    print_number: int,
    q_table: np.ndarray | None,
    epsilon: float,
    session_filename: str,
    octoprint_url: str,
    api_key: str,
    *,
    print_angle_deg: float,
    open_loop: bool = False,
    capture_layer_photos: bool = LAYER_PHOTO_ENABLED,
    layer_labels: LayerLabelMap | None = None,
) -> tuple[np.ndarray | None, int, float, int, str]:
    control_mode = "open_loop" if open_loop else "rl"
    run_folder = print_run_folder_name(
        print_number,
        control_mode=control_mode,
        print_angle_deg=print_angle_deg,
    )
    watcher = LayerChangeWatcher(octoprint_url, api_key)
    watcher.start()

    base_speed, base_flow, base_fan = base_print_settings()
    meta_path = write_print_run_metadata(
        print_number,
        print_angle_deg=print_angle_deg,
        control_mode=control_mode,
        session_filename=session_filename,
    )

    print(f"\n{'='*72}")
    mode_text = "open-loop logging" if open_loop else "layer-triggered RL"
    print(
        f"  PRINT #{print_number:>3}  |  angle={print_angle_deg:.1f}° from vertical  |  "
        f"{mode_text}  |  {datetime.now().strftime('%H:%M:%S')}"
    )
    if open_loop:
        print(
            f"  Controls: {base_speed:.0f} mm/s, {base_flow:.0f}% flow, "
            f"{base_fan:.0f}% fan (fixed, no CNN, no RL)"
        )
    else:
        print(f"  Q-table: {session_filename}  |  explore={epsilon:.0%}")
    print(f"  Run meta: {meta_path}")
    if capture_layer_photos:
        print(f"  Run folder: {run_folder}/")
        print(
            f"  Photos: {print_run_photos_dir(print_number, control_mode=control_mode, print_angle_deg=print_angle_deg)}/"
        )
    if layer_labels:
        print(f"  Labels: {layer_labels.describe()}")
    print(f"{'='*72}")
    print(
        f"{'Layer':>6}  {'DepC':>7}  {'Speed':>7}  {'Flow':>5}  {'Fan':>4}  "
        f"{'Visual':>9}  {'Action':>10}  {'Reward':>7}  {'Window':>7}  How"
    )
    print("-" * 96)

    log_file, log_writer, log_filename = create_log_file(
        print_number,
        session_filename,
        print_angle_deg=print_angle_deg,
        control_mode=control_mode,
    )
    if open_loop:
        current_speed_index, current_flow_index, current_fan_index = base_control_indices()
        new_speed = base_speed
        new_flow = base_flow
        new_fan = base_fan
    else:
        current_speed_index = len(SPEEDS) // 2
        current_flow_index = int(np.argmin(np.abs(FLOWS - BASE_FLOW_PERCENT)))
        current_fan_index = int(np.argmin(np.abs(FANS - BASE_FAN_PERCENT)))
        new_speed = float(SPEEDS[current_speed_index])
        new_flow = float(FLOWS[current_flow_index])
        new_fan = float(FANS[current_fan_index])
    speed_publisher = RlSpeedStatusPublisher()
    speed_publisher.start(
        print_number,
        new_speed,
        initial_flow_percent=new_flow,
        initial_fan_percent=new_fan,
    )
    send_print_controls(new_speed, new_flow, new_fan)
    total_reward = 0.0
    layers_in_window = 0
    layers_completed = 0
    rng = np.random.default_rng()
    action_names = [name for _param, _delta, name in ACTION_SPECS]
    open_loop_action_name = "base fixed"

    try:
        watcher.wait_for_print_start()

        while True:
            layer = watcher.wait_for_layer_complete()
            if layer is None:
                print("\n  Print finished — ending episode.")
                break

            layers_completed += 1

            layer_photo: str | None = None
            visual = assess_layer_visual(
                layer,
                None,
                manual_labels=layer_labels,
            )
            if capture_layer_photos:
                layer_photo = capture_layer_photo(
                    octoprint_url,
                    print_number,
                    layer,
                    control_mode=control_mode,
                    print_angle_deg=print_angle_deg,
                )
                visual = assess_layer_visual(
                    layer,
                    layer_photo,
                    manual_labels=layer_labels,
                )

            obs = read_thermal()
            temps = read_printer_temps()
            obs_temp = obs["deposit_temp_c"]
            current_state = discretise(
                obs_temp,
                current_speed_index,
                current_flow_index,
                current_fan_index,
                visual,
            )

            if open_loop:
                chosen_action = -1
                how = "open_loop"
                action_label = open_loop_action_name
                new_speed = base_speed
                new_flow = base_flow
                new_fan = base_fan
            else:
                if rng.random() < epsilon:
                    chosen_action = int(rng.integers(N_ACTIONS))
                    how = "explore"
                else:
                    chosen_action = int(np.argmax(q_table[current_state]))
                    how = "exploit"

                current_speed_index, current_flow_index, current_fan_index = apply_action(
                    chosen_action,
                    current_speed_index,
                    current_flow_index,
                    current_fan_index,
                )
                new_speed = float(SPEEDS[current_speed_index])
                new_flow = float(FLOWS[current_flow_index])
                new_fan = float(FANS[current_fan_index])
                action_label = action_names[chosen_action].strip()
            send_print_controls(new_speed, new_flow, new_fan)
            speed_publisher.update(
                new_speed,
                layer,
                flow_percent=new_flow,
                fan_percent=new_fan,
                visual_defect=visual.label,
            )

            time.sleep(SPEED_SETTLE_TIME)
            new_obs = read_thermal()
            new_temps = read_printer_temps()
            new_temp = new_obs["deposit_temp_c"]
            new_state = discretise(
                new_temp,
                current_speed_index,
                current_flow_index,
                current_fan_index,
                visual,
            )
            reward = calculate_reward(new_temp, visual) if not open_loop else 0.0

            current_q_value = 0.0
            q_value_after = 0.0
            if not open_loop and q_table is not None:
                current_q_value = float(q_table[current_state, chosen_action])
                best_future_q = float(np.max(q_table[new_state]))
                q_table[current_state, chosen_action] = current_q_value + ALPHA * (
                    reward + GAMMA * best_future_q - current_q_value
                )
                q_value_after = float(q_table[current_state, chosen_action])
                save_q_table(q_table, session_filename)

            in_window = BAND_LO <= new_temp <= BAND_HI
            if in_window:
                layers_in_window += 1
            total_reward += reward

            flag = "YES  v" if in_window else "NO   x"
            visual_text = (visual.label or "unknown")[:9]
            print(
                f"{layer:>6}  {new_temp:>6.1f}C  {new_speed:>5.0f}mm/s  "
                f"{new_flow:>4.0f}%  {new_fan:>3.0f}%  {visual_text:>9}  "
                f"{action_label:>10}  {reward:>7.2f}  {flag:>7}  {how}"
            )

            log_writer.writerow([
                print_number,
                round(print_angle_deg, 2),
                control_mode,
                layer,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                round(new_speed, 2),
                round(new_flow, 2),
                round(new_fan, 2),
                round(obs["deposit_temp_c"], 2),
                round(obs["roi_max"], 2),
                round(obs["roi_mean"], 2),
                round(obs["roi_std"], 2),
                round(obs["roi_min"], 2),
                round(obs["frame_max"], 2),
                round(obs["frame_mean"], 2),
                temps["hotend_actual"],
                temps["hotend_target"],
                temps["bed_actual"],
                temps["bed_target"],
                obs["roi_skip_top"],
                obs["roi_n_rows"],
                current_state,
                chosen_action,
                action_label,
                how,
                round(new_obs["roi_max"], 2),
                round(new_obs["roi_mean"], 2),
                round(new_obs["frame_max"], 2),
                new_state,
                round(reward, 4),
                in_window,
                round(current_q_value, 4),
                round(q_value_after, 4),
                round(epsilon, 3),
                os.path.basename(session_filename),
                layer_photo or "",
                visual.label or "",
                "" if visual.confidence is None else round(visual.confidence, 4),
                visual.source or "",
            ])
            log_file.flush()

    except KeyboardInterrupt:
        print("\n  Episode interrupted.")
    finally:
        speed_publisher.stop()
        watcher.stop()
        log_file.close()

    if layers_completed == 0:
        print("  No layers completed.")
        return q_table, 0, 0.0, 0, log_filename

    pct = layers_in_window / layers_completed
    if open_loop:
        print(
            f"\n  Print #{print_number} done — angle={print_angle_deg:.1f}°  "
            f"{layers_completed} layer(s) logged at base settings"
        )
    else:
        print(
            f"\n  Print #{print_number} done — angle={print_angle_deg:.1f}°  "
            f"{layers_in_window}/{layers_completed} layers in window ({pct:.0%})  "
            f"|  total reward: {total_reward:.1f}"
        )
    print(f"  Log : {log_filename}")
    if not open_loop:
        print(f"  Q   : {session_filename}")
    return q_table, layers_in_window, total_reward, layers_completed, log_filename


# ══════════════════════════════════════════════════════════════════
#  SECTION 6 — MAIN
# ══════════════════════════════════════════════════════════════════

def prompt_for_print_angle(suggested: float | None = None) -> float:
    hint = f" [{suggested:.1f}]" if suggested is not None else ""
    while True:
        answer = input(
            f"  Overhang angle from vertical (degrees){hint}: "
        ).strip()
        if not answer:
            if suggested is not None:
                return suggested
            print("  Enter a numeric angle in degrees (e.g. 30).")
            continue
        try:
            angle = float(answer)
        except ValueError:
            print("  Invalid angle — enter a number like 30 or 45.5.")
            continue
        if angle < 0.0 or angle > 90.0:
            print("  Angle should be between 0° (vertical) and 90° (horizontal).")
            continue
        return angle


def prompt_for_layer_labels(
    print_number: int,
    suggested: LayerLabelMap | None = None,
) -> LayerLabelMap:
    hint = suggested.describe() if suggested else ""
    if hint and hint != "(none)":
        prompt = (
            f"  Layer labels for print #{print_number} "
            f"(e.g. 1-30:normal,31-50:defective) [{hint}]: "
        )
    else:
        prompt = (
            f"  Layer labels for print #{print_number} "
            "(e.g. 1-30:normal,31-50:defective) or Enter to skip: "
        )
    answer = input(prompt).strip()
    if not answer:
        return suggested or LayerLabelMap()
    return LayerLabelMap.parse(answer)


def main() -> int:
    global DEBUG_LAYERS, LAYER_PHOTO_ENABLED

    parser = argparse.ArgumentParser(description="Layer-triggered RL logger")
    parser.add_argument(
        "--debug-layers",
        action="store_true",
        help="Print layer-detection diagnostics",
    )
    parser.add_argument(
        "--no-layer-photos",
        action="store_true",
        help="Skip per-layer webcam snapshots",
    )
    parser.add_argument(
        "--layer-labels",
        metavar="SPEC",
        help='Manual labels for all prints, e.g. "1-30:normal,31-50:defective"',
    )
    parser.add_argument(
        "--layer-labels-file",
        metavar="PATH",
        help="JSON file with per-print layer label ranges (keys = print folder names)",
    )
    parser.add_argument(
        "--apply-labels-to",
        metavar="CSV",
        help="Update visual_defect columns in an existing log CSV and exit",
    )
    parser.add_argument(
        "--no-label-prompt",
        action="store_true",
        help="Do not ask for layer labels before each print",
    )
    parser.add_argument(
        "--no-cnn",
        action="store_true",
        help="Disable CNN inference for layer photos",
    )
    parser.add_argument(
        "--cnn-model",
        metavar="PATH",
        default=str(LAYER_DEFECT_MODEL_PATH),
        help="Path to the trained layer defect CNN weights",
    )
    parser.add_argument(
        "--cnn-min-confidence",
        type=float,
        default=CNN_MIN_CONFIDENCE,
        help="Minimum softmax confidence required to accept a CNN label",
    )
    parser.add_argument(
        "--cnn-device",
        choices=("cpu", "cuda"),
        help="Device for CNN inference (default: cuda when available)",
    )
    parser.add_argument(
        "--apply-cnn-to",
        metavar="CSV",
        help="Fill visual_defect columns in an existing log using the CNN and exit",
    )
    parser.add_argument(
        "--print-angle",
        type=float,
        metavar="DEG",
        help="Overhang angle from vertical in degrees (prompted per print if omitted)",
    )
    parser.add_argument(
        "--open-loop",
        action="store_true",
        help=(
            "Baseline logging only: fixed 170 mm/s, 100%% flow, 100%% fan; "
            "no CNN and no RL control"
        ),
    )
    args = parser.parse_args()
    DEBUG_LAYERS = args.debug_layers
    LAYER_PHOTO_ENABLED = not args.no_layer_photos
    open_loop = args.open_loop
    if open_loop:
        args.no_cnn = True
    if args.print_angle is not None and (
        args.print_angle < 0.0 or args.print_angle > 90.0
    ):
        print("--print-angle must be between 0 and 90 degrees.")
        return 1
    configure_cnn(
        enabled=not args.no_cnn,
        model_path=args.cnn_model,
        min_confidence=args.cnn_min_confidence,
        device=args.cnn_device,
    )

    if args.apply_cnn_to:
        if args.no_cnn:
            print("--apply-cnn-to cannot be combined with --no-cnn.")
            return 1
        try:
            from layer_defect_cnn import apply_cnn_to_csv

            updated = apply_cnn_to_csv(
                args.apply_cnn_to,
                model_path=args.cnn_model,
                min_confidence=args.cnn_min_confidence,
                device=args.cnn_device,
            )
        except (OSError, ValueError, FileNotFoundError) as error:
            print(f"Failed to apply CNN labels: {error}")
            return 1
        print(f"Updated {updated} row(s) in {args.apply_cnn_to}")
        return 0

    if args.apply_labels_to:
        if not args.layer_labels:
            print("--apply-labels-to requires --layer-labels.")
            return 1
        try:
            label_map = LayerLabelMap.parse(args.layer_labels)
            updated = apply_layer_labels_to_csv(args.apply_labels_to, label_map)
        except (OSError, ValueError) as error:
            print(f"Failed to apply labels: {error}")
            return 1
        print(f"Updated {updated} row(s) in {args.apply_labels_to}")
        return 0

    try:
        label_session = LayerLabelSession.from_specs(
            default_spec=args.layer_labels,
            labels_file=args.layer_labels_file,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Invalid layer label configuration: {error}")
        return 1

    api_key = load_octoprint_api_key()
    if not api_key:
        print("Set PRUSASAM_OCTOPRINT_API_KEY or create api_key.py for layer detection.")
        return 1

    print("\nRL Layer Logger — maximum overhang angle study")
    if open_loop:
        print(
            "Open-loop mode: fixed base settings, no CNN, no RL control.\n"
            "Uses dashboard_bridge for IR temperature, G-code overrides, and layer photos.\n"
        )
    else:
        print(
            "RL mode: Q-learning with automatic layer detection.\n"
            "Uses dashboard_bridge for IR temperature, G-code overrides, and layer photos.\n"
        )

    if not test_connections():
        return 1

    check = input("Inspect thermal frame to verify ROI settings? (y/n): ")
    if check.strip().lower() == "y":
        inspect_thermal_frame()
        input(
            "\nAdjust PRUSASAM_ROI_SKIP_TOP_MIN / PRUSASAM_ROI_N_ROWS / PRUSASAM_ROI_N_COLS / "
            "PRUSASAM_HOTEND_PIXEL_MAX_C if needed. Press Enter..."
        )

    session_filename = ""
    q_table: np.ndarray | None = None
    if not open_loop:
        session_filename = get_session_filename()
        print(f"\nThis session writes to:\n  {session_filename}\n")
        q_table = load_q_table()
    octoprint_url = read_printer_temps()["octoprint_url"] or DEFAULT_OCTOPRINT_URL

    all_pct: list[float] = []
    default_print_angle = args.print_angle
    try:
        for print_number in range(1, N_PRINTS + 1):
            epsilon = max(
                EPSILON_MIN,
                EPSILON_START - print_number / (0.7 * N_PRINTS),
            )
            print(f"\nReady for print #{print_number}.")
            print_angle_deg = prompt_for_print_angle(default_print_angle)
            default_print_angle = print_angle_deg
            print_labels = label_session.for_print(
                print_number,
                control_mode="open_loop" if open_loop else "rl",
                print_angle_deg=print_angle_deg,
            )
            if not args.no_label_prompt:
                print_labels = prompt_for_layer_labels(print_number, print_labels)
            input("  Press Enter when the print is running...")
            q_table, layers_in_window, _reward, layers_completed, log_filename = run_one_print(
                print_number,
                q_table,
                epsilon,
                session_filename,
                octoprint_url,
                api_key,
                print_angle_deg=print_angle_deg,
                open_loop=open_loop,
                capture_layer_photos=LAYER_PHOTO_ENABLED,
                layer_labels=print_labels,
            )
            if not args.no_label_prompt and print_labels:
                revise = input(
                    "  Revise layer labels for this print after review? (y/n): "
                ).strip().lower()
                if revise == "y":
                    revised_labels = prompt_for_layer_labels(print_number, print_labels)
                    if revised_labels:
                        updated = apply_layer_labels_to_csv(log_filename, revised_labels)
                        print(f"  Updated {updated} row(s) in {log_filename}")
            if layers_completed > 0:
                all_pct.append(layers_in_window / layers_completed)
            if print_number < N_PRINTS:
                answer = input(f"\n  Continue to print #{print_number + 1}? (y/n): ")
                if answer.strip().lower() != "y":
                    break
    except KeyboardInterrupt:
        if open_loop:
            print("\nStopped by user.")
        else:
            print("\nStopped by user. Q-table saved after each layer.")

    if all_pct:
        print(f"\nSession summary — {len(all_pct)} print(s)")
        if open_loop:
            print("  Open-loop runs logged with fixed base settings.")
        else:
            print(f"  Average in window : {np.mean(all_pct):.0%}")
            print(f"  Best print        : {max(all_pct):.0%}")
            print(f"  Q-table           : {session_filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
