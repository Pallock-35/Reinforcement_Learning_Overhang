from __future__ import annotations

import json
import os
import socket
import struct
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

try:
    from api_key import API_key
except ModuleNotFoundError:
    API_key = os.environ.get("PRUSASAM_OCTOPRINT_API_KEY", "")

from thermal_roi import attach_roi_to_thermal_snapshot

HOST = os.environ.get("PRUSASAM_BRIDGE_HOST", "127.0.0.1")
PORT = 8765
ROOT = Path(__file__).resolve().parent
HTML_PATH = ROOT / "prusa_dashboard.html"
OCTOPRINT_URL = "http://10.105.77.241/"
OCTOPRINT_HOST = urlparse(OCTOPRINT_URL).hostname or "127.0.0.1"
THERMAL_HOST = os.environ.get("PRUSASAM_THERMAL_HOST", OCTOPRINT_HOST)
THERMAL_PORT = int(os.environ.get("PRUSASAM_THERMAL_PORT", "5001"))
THERMAL_ROWS = 62
THERMAL_COLS = 80
THERMAL_FRAME_FLOATS = THERMAL_ROWS * THERMAL_COLS
THERMAL_FRAME_BYTES = THERMAL_FRAME_FLOATS * 4
BASE_SPEED_MMS = float(os.environ.get("PRUSASAM_BASE_SPEED_MMS", "170"))

USEFUL_TEMPERATURE_KEYS = ("bed", "tool0")

THERMAL_FEED: ThermalMatrixFeed | None = None

_RL_LOCK = threading.Lock()
_RL_STATUS: dict[str, Any] | None = None
RL_STATUS_MAX_AGE_S = float(os.environ.get("PRUSASAM_RL_STATUS_MAX_AGE_S", "20"))

_THERMAL_CONTROL_LOCK = threading.Lock()
_THERMAL_CONTROL_STATUS: dict[str, Any] | None = None
THERMAL_CONTROL_STATUS_MAX_AGE_S = float(
    os.environ.get("PRUSASAM_THERMAL_CONTROL_STATUS_MAX_AGE_S", "20")
)


def _read_json_response(response: Any) -> dict[str, Any]:
    body = response.read()
    if not body:
        return {}
    return json.loads(body.decode("utf-8"))


def _fetch_json(url: str, api_key: str) -> dict[str, Any]:
    request = Request(url, headers={"X-Api-Key": api_key})

    with urlopen(request, timeout=10) as response:
        return _read_json_response(response)


def fetch_printer_status(base_url: str, api_key: str) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/printer"
    return _fetch_json(url, api_key)


def send_gcode_commands(base_url: str, api_key: str, commands: list[str]) -> int:
    url = f"{base_url.rstrip('/')}/api/printer/command"
    body_key = "command" if len(commands) == 1 else "commands"
    body_value: str | list[str] = commands[0] if len(commands) == 1 else commands
    request = Request(
        url,
        data=json.dumps({body_key: body_value}).encode("utf-8"),
        headers={
            "X-Api-Key": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urlopen(request, timeout=10) as response:
        return response.getcode()


def _parse_temperature(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _temperature_reading(sensor_data: dict[str, Any] | None) -> dict[str, float | None]:
    if not isinstance(sensor_data, dict):
        return {"actual": None, "target": None, "offset": None}
    return {
        "actual": _parse_temperature(sensor_data.get("actual")),
        "target": _parse_temperature(sensor_data.get("target")),
        "offset": _parse_temperature(sensor_data.get("offset")),
    }


def _format_matrix_rows(values: list[float]) -> list[list[float]]:
    return [values[index : index + THERMAL_COLS] for index in range(0, len(values), THERMAL_COLS)]


def _extract_temperature_readings(payload: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    temperature_payload = payload.get("temperature") or {}
    temperatures: dict[str, dict[str, float | None]] = {}
    if not isinstance(temperature_payload, dict):
        return temperatures
    for sensor_name in USEFUL_TEMPERATURE_KEYS:
        temperatures[sensor_name] = _temperature_reading(temperature_payload.get(sensor_name))
    return temperatures


def _extract_feedrate_percent(payload: dict[str, Any]) -> float | None:
    state = payload.get("state")
    if isinstance(state, dict):
        flags = state.get("flags") or {}
        feedrate = flags.get("feedrate")
        if feedrate is not None:
            return float(feedrate)
    feedrate = payload.get("feedrate")
    if feedrate is not None:
        return float(feedrate)
    return None


def _speed_from_feedrate(feedrate_percent: float | None, base_speed_mms: float) -> float | None:
    if feedrate_percent is None:
        return None
    return base_speed_mms * (feedrate_percent / 100.0)


def build_speed_snapshot(octoprint_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    thermal_status = read_thermal_control_status()
    if thermal_status is not None:
        speed_mms = thermal_status.get("speed_mms")
        base_speed = float(thermal_status.get("base_speed_mms") or BASE_SPEED_MMS)
        feedrate_percent = None
        if speed_mms is not None and base_speed > 0:
            feedrate_percent = (float(speed_mms) / base_speed) * 100.0
        return {
            "source": "thermal_layer_logger",
            "speed_mms": float(speed_mms) if speed_mms is not None else None,
            "feedrate_percent": feedrate_percent,
            "base_speed_mms": base_speed,
            "print_number": thermal_status.get("print_number"),
            "layer": thermal_status.get("layer"),
            "rl_active": True,
        }

    rl_status = read_rl_layer_logger_status()
    if rl_status is not None:
        speed_mms = rl_status.get("speed_mms")
        base_speed = float(rl_status.get("base_speed_mms") or BASE_SPEED_MMS)
        feedrate_percent = None
        if speed_mms is not None and base_speed > 0:
            feedrate_percent = (float(speed_mms) / base_speed) * 100.0
        return {
            "source": "rl_layer_logger",
            "speed_mms": float(speed_mms) if speed_mms is not None else None,
            "feedrate_percent": feedrate_percent,
            "base_speed_mms": base_speed,
            "print_number": rl_status.get("print_number"),
            "layer": rl_status.get("layer"),
            "rl_active": True,
        }

    feedrate_percent = _extract_feedrate_percent(octoprint_payload or {})
    return {
        "source": "octoprint",
        "speed_mms": _speed_from_feedrate(feedrate_percent, BASE_SPEED_MMS),
        "feedrate_percent": feedrate_percent,
        "base_speed_mms": BASE_SPEED_MMS,
        "print_number": None,
        "layer": None,
        "rl_active": False,
    }


def update_rl_layer_logger_status(payload: dict[str, Any]) -> dict[str, Any]:
    global _RL_STATUS

    active = bool(payload.get("active", True))
    if not active:
        with _RL_LOCK:
            _RL_STATUS = None
        return {"active": False, "source": "rl_layer_logger"}

    speed_mms = payload.get("speed_mms")
    if speed_mms is None:
        raise ValueError("speed_mms is required when rl status is active.")

    status = {
        "active": True,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "speed_mms": float(speed_mms),
        "base_speed_mms": float(payload.get("base_speed_mms") or BASE_SPEED_MMS),
        "print_number": payload.get("print_number"),
        "layer": payload.get("layer"),
        "source": "rl_layer_logger",
    }

    with _RL_LOCK:
        _RL_STATUS = status
    return status


def read_rl_layer_logger_status() -> dict[str, Any] | None:
    with _RL_LOCK:
        if _RL_STATUS is None or not _RL_STATUS.get("active"):
            return None
        status = dict(_RL_STATUS)

    updated_at = status.get("updated_at")
    if not updated_at:
        return None
    try:
        updated_time = datetime.fromisoformat(str(updated_at))
    except ValueError:
        return None
    age_s = (datetime.now() - updated_time).total_seconds()
    if age_s > RL_STATUS_MAX_AGE_S:
        return None
    return status


def update_thermal_control_status(payload: dict[str, Any]) -> dict[str, Any]:
    global _THERMAL_CONTROL_STATUS

    active = bool(payload.get("active", True))
    if not active:
        with _THERMAL_CONTROL_LOCK:
            _THERMAL_CONTROL_STATUS = None
        return {"active": False, "source": "thermal_layer_logger"}

    speed_mms = payload.get("speed_mms")
    if speed_mms is None:
        raise ValueError("speed_mms is required when thermal control status is active.")

    nozzle_temp_c = payload.get("nozzle_temp_c")
    status = {
        "active": True,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "control_mode": payload.get("control_mode"),
        "speed_mms": float(speed_mms),
        "nozzle_temp_c": float(nozzle_temp_c) if nozzle_temp_c is not None else None,
        "fan_normalized": payload.get("fan_normalized", 1.0),
        "dwell_s": payload.get("dwell_s", 0.0),
        "base_speed_mms": float(payload.get("base_speed_mms") or BASE_SPEED_MMS),
        "print_number": payload.get("print_number"),
        "layer": payload.get("layer"),
        "T_k": payload.get("T_k"),
        "D_k": payload.get("D_k"),
        "mpc_cost": payload.get("mpc_cost"),
        "mpc_success": payload.get("mpc_success"),
        "source": "thermal_layer_logger",
    }

    with _THERMAL_CONTROL_LOCK:
        _THERMAL_CONTROL_STATUS = status
    return status


def read_thermal_control_status() -> dict[str, Any] | None:
    with _THERMAL_CONTROL_LOCK:
        if _THERMAL_CONTROL_STATUS is None or not _THERMAL_CONTROL_STATUS.get("active"):
            return None
        status = dict(_THERMAL_CONTROL_STATUS)

    updated_at = status.get("updated_at")
    if not updated_at:
        return None
    try:
        updated_time = datetime.fromisoformat(str(updated_at))
    except ValueError:
        return None
    age_s = (datetime.now() - updated_time).total_seconds()
    if age_s > THERMAL_CONTROL_STATUS_MAX_AGE_S:
        return None
    return status


def _enriched_thermal_snapshot() -> dict[str, Any] | None:
    if THERMAL_FEED is None:
        return None
    return attach_roi_to_thermal_snapshot(THERMAL_FEED.snapshot())


def _current_thermal_snapshot() -> tuple[dict[str, Any] | None, str]:
    if THERMAL_FEED is None:
        return None, "Thermal feed not configured"
    snapshot = _enriched_thermal_snapshot()
    if snapshot is None:
        return None, "no_frame"
    return snapshot, ""


def _normalize_gcode_payload(payload: dict[str, Any]) -> list[str]:
    has_command = "command" in payload
    has_commands = "commands" in payload
    if has_command == has_commands:
        raise ValueError("Provide exactly one of 'command' or 'commands'.")

    raw_commands = [payload["command"]] if has_command else payload["commands"]
    if not isinstance(raw_commands, list):
        raise ValueError("'commands' must be a list of G-code strings.")

    commands: list[str] = []
    for raw_command in raw_commands:
        if not isinstance(raw_command, str):
            raise ValueError("Each G-code command must be a string.")
        command = raw_command.strip()
        if not command:
            raise ValueError("G-code commands must not be empty.")
        commands.append(command)
    if not commands:
        raise ValueError("Provide at least one G-code command.")
    return commands


def build_temperature_snapshot() -> dict[str, Any]:
    payload = fetch_printer_status(OCTOPRINT_URL, API_key)
    temperatures = _extract_temperature_readings(payload)
    return {
        "ok": True,
        "source": "octoprint",
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "temperatures": temperatures,
        "bed_temperature": temperatures.get("bed", {"actual": None, "target": None, "offset": None}),
        "octoprint_url": OCTOPRINT_URL,
    }


def build_telemetry_snapshot() -> dict[str, Any]:
    temperatures: dict[str, dict[str, float | None]] = {}
    errors: dict[str, str] = {}
    try:
        temperatures = build_temperature_snapshot()["temperatures"]
    except Exception as error:  # noqa: BLE001 - return partial telemetry when one source is unavailable
        errors["temperatures"] = str(error)

    thermal, thermal_error = _current_thermal_snapshot()
    if thermal_error:
        errors["thermal"] = thermal_error

    return {
        "ok": not errors,
        "source": "python",
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "bed_temperature": temperatures.get("bed", {"actual": None, "target": None, "offset": None}),
        "temperatures": temperatures,
        "thermal": thermal,
        "thermal_matrix": thermal["values"] if thermal else None,
        "thermal_rows": thermal["rows"] if thermal else THERMAL_ROWS,
        "thermal_cols": thermal["cols"] if thermal else THERMAL_COLS,
        "errors": errors,
        "octoprint_url": OCTOPRINT_URL,
    }


class ThermalMatrixFeed:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._connected = False
        self._last_error = ""
        self._updated_at: datetime | None = None
        self._min_value: float | None = None
        self._max_value: float | None = None
        self._matrix: list[list[float]] | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)

    def snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            if self._matrix is None:
                return None
            return {
                "source": "thermal-pi",
                "connected": self._connected,
                "updated_at": self._updated_at.strftime("%Y-%m-%d %H:%M:%S") if self._updated_at else "",
                "error_text": self._last_error,
                "rows": THERMAL_ROWS,
                "cols": THERMAL_COLS,
                "min": self._min_value,
                "max": self._max_value,
                "values": self._matrix,
            }

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                with socket.create_connection((self.host, self.port), timeout=10) as conn:
                    conn.settimeout(5.0)
                    with self._lock:
                        self._connected = True
                        self._last_error = ""
                    while not self._stop.is_set():
                        header = self._recv_exact(conn, 4)
                        if not header:
                            raise EOFError("Thermal stream closed")
                        payload_length = struct.unpack("<I", header)[0]
                        if payload_length != THERMAL_FRAME_BYTES:
                            self._recv_exact(conn, payload_length)
                            raise ValueError(f"Unexpected thermal frame size: {payload_length}")
                        payload = self._recv_exact(conn, payload_length)
                        values = list(struct.unpack(f"<{THERMAL_FRAME_FLOATS}f", payload))
                        now = datetime.now()
                        with self._lock:
                            self._connected = True
                            self._last_error = ""
                            self._updated_at = now
                            self._min_value = min(values)
                            self._max_value = max(values)
                            self._matrix = _format_matrix_rows(values)
            except Exception as error:  # noqa: BLE001 - keep the thermal feed resilient to disconnects
                with self._lock:
                    self._connected = False
                    self._last_error = str(error)
                if self._stop.wait(1.0):
                    break

    @staticmethod
    def _recv_exact(conn: socket.socket, size: int) -> bytes:
        buffer = bytearray()
        while len(buffer) < size:
            chunk = conn.recv(size - len(buffer))
            if not chunk:
                raise EOFError("Socket closed while reading thermal data")
            buffer.extend(chunk)
        return bytes(buffer)


def build_dashboard_snapshot() -> dict[str, Any]:
    payload = fetch_printer_status(OCTOPRINT_URL, API_key)

    state = payload.get("state")
    state_text = "State: unknown"
    is_printing = False
    if isinstance(state, dict):
        state_text = f"State: {state.get('text', 'unknown')}"
        flags = state.get("flags") or {}
        is_printing = bool(flags.get("printing", False)) or ("print" in str(state.get("text", "")).lower())

    temperature_payload = payload.get("temperature") or {}
    temperatures: dict[str, dict[str, float | None]] = {}
    temperature_lines: list[str] = []
    if isinstance(temperature_payload, dict):
        for sensor_name in USEFUL_TEMPERATURE_KEYS:
            reading = _temperature_reading(temperature_payload.get(sensor_name))
            temperatures[sensor_name] = reading
            temperature_lines.append(
                f"{sensor_name}: actual={reading['actual']}, target={reading['target']}, offset={reading['offset']}"
            )

    thermal_frame = _enriched_thermal_snapshot()
    thermal_summary = "Thermal feed is unavailable."
    if thermal_frame is not None:
        thermal_summary = "Thermal feed is live." if thermal_frame["connected"] else f"Thermal feed reconnecting: {thermal_frame['error_text']}"

    sample_time = datetime.now()
    return {
        "source": "python",
        "connected": True,
        "mode": "live",
        "updated_at": sample_time.strftime("%Y-%m-%d %H:%M:%S"),
        "state_text": state_text,
        "is_printing": is_printing,
        "error_text": "",
        "temperatures": temperatures,
        "temperature_lines": temperature_lines,
        "speed": build_speed_snapshot(payload),
        "thermal_frame": thermal_frame,
        "thermal_summary": thermal_summary,
        "webcam_url": f"{OCTOPRINT_URL.rstrip('/')}/webcam/?action=stream",
        "connection_summary": f"Connected to {OCTOPRINT_URL}",
        "octoprint_url": OCTOPRINT_URL,
    }


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "PrusaSAMBridge/1.0"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler interface
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            self._serve_html()
            return
        if path == "/api/health":
            self._serve_json({"ok": True, "time": datetime.now().isoformat(timespec="seconds")})
            return
        if path == "/api/thermal":
            self._serve_thermal()
            return
        if path == "/api/temperatures":
            self._serve_temperatures()
            return
        if path == "/api/telemetry":
            self._serve_telemetry()
            return
        if path == "/api/dashboard":
            self._serve_dashboard()
            return
        if path == "/api/rl-status":
            self._serve_rl_status_get()
            return
        if path == "/api/thermal-control-status":
            self._serve_thermal_control_status_get()
            return
        self.send_error(404, "Not found")

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler interface
        path = urlparse(self.path).path
        if path == "/api/gcode":
            self._serve_gcode()
            return
        if path == "/api/rl-status":
            self._serve_rl_status_post()
            return
        if path == "/api/thermal-control-status":
            self._serve_thermal_control_status_post()
            return
        self.send_error(404, "Not found")

    def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler interface
        self.send_response(204)
        self._set_cors_headers()
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003 - match base class signature
        return

    def _serve_dashboard(self) -> None:
        try:
            snapshot = build_dashboard_snapshot()
        except Exception as error:  # noqa: BLE001 - surface the live connection error to the UI
            snapshot = {
                "source": "python",
                "connected": False,
                "mode": "offline",
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "state_text": "State: unavailable",
                "is_printing": False,
                "error_text": str(error),
                "temperatures": {sensor_name: {"actual": None, "target": None, "offset": None} for sensor_name in USEFUL_TEMPERATURE_KEYS},
                "temperature_lines": [],
                "speed": build_speed_snapshot(None),
                "thermal_frame": _enriched_thermal_snapshot(),
                "thermal_summary": "Thermal feed is unavailable.",
                "webcam_url": "",
                "connection_summary": "Python bridge could not reach OctoPrint.",
                "octoprint_url": OCTOPRINT_URL,
            }
        self._serve_json(snapshot)

    def _serve_html(self) -> None:
        if HTML_PATH.exists():
            content = HTML_PATH.read_text(encoding="utf-8")
        else:
            content = "<html><body><h1>PrusaSAM dashboard</h1><p>prusa_dashboard.html is missing.</p></body></html>"
        body = content.encode("utf-8")
        self.send_response(200)
        self._set_cors_headers("text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_json(self, payload: dict[str, Any], status_code: int = 200) -> None:
        body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status_code)
        self._set_cors_headers("application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_thermal(self) -> None:
        try:
            if THERMAL_FEED is None:
                raise RuntimeError("Thermal feed not configured")
            snapshot = _enriched_thermal_snapshot()
            if snapshot is None:
                self._serve_json({"ok": False, "error": "no_frame"})
                return
            self._serve_json({"ok": True, "thermal": snapshot})
        except Exception as error:
            self._serve_json({"ok": False, "error": str(error)})

    def _serve_temperatures(self) -> None:
        try:
            self._serve_json(build_temperature_snapshot())
        except Exception as error:
            self._serve_json({"ok": False, "error": str(error)}, status_code=502)

    def _serve_telemetry(self) -> None:
        self._serve_json(build_telemetry_snapshot())

    def _serve_gcode(self) -> None:
        try:
            payload = self._read_json_body()
            commands = _normalize_gcode_payload(payload)
        except ValueError as error:
            self._serve_json({"ok": False, "error": str(error)}, status_code=400)
            return

        try:
            octoprint_status = send_gcode_commands(OCTOPRINT_URL, API_key, commands)
        except Exception as error:
            self._serve_json({"ok": False, "error": str(error)}, status_code=502)
            return

        self._serve_json({"ok": True, "sent": commands, "octoprint_status": octoprint_status})

    def _serve_rl_status_get(self) -> None:
        status = read_rl_layer_logger_status()
        self._serve_json({"ok": True, "active": status is not None, "status": status})

    def _serve_rl_status_post(self) -> None:
        try:
            payload = self._read_json_body()
            status = update_rl_layer_logger_status(payload)
        except ValueError as error:
            self._serve_json({"ok": False, "error": str(error)}, status_code=400)
            return
        self._serve_json({"ok": True, "status": status})

    def _serve_thermal_control_status_get(self) -> None:
        status = read_thermal_control_status()
        self._serve_json({"ok": True, "active": status is not None, "status": status})

    def _serve_thermal_control_status_post(self) -> None:
        try:
            payload = self._read_json_body()
            status = update_thermal_control_status(payload)
        except ValueError as error:
            self._serve_json({"ok": False, "error": str(error)}, status_code=400)
            return
        self._serve_json({"ok": True, "status": status})

    def _read_json_body(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            content_length = int(raw_length)
        except ValueError as error:
            raise ValueError("Invalid Content-Length header.") from error
        if content_length <= 0:
            raise ValueError("Request body must contain JSON.")
        if content_length > 64 * 1024:
            raise ValueError("Request body is too large.")

        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("Request body must be valid JSON.") from error
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    def _set_cors_headers(self, content_type: str | None = None) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Requested-With, X-Api-Key")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        if content_type is not None:
            self.send_header("Content-Type", content_type)


def main() -> int:
    global THERMAL_FEED
    THERMAL_FEED = ThermalMatrixFeed(THERMAL_HOST, THERMAL_PORT)
    THERMAL_FEED.start()
    server = ThreadingHTTPServer((HOST, PORT), DashboardHandler)
    print(f"Serving PrusaSAM dashboard bridge on http://{HOST}:{PORT}")
    print(
        f"Open the dashboard at http://127.0.0.1:{PORT}/ or open {HTML_PATH.name} and keep its API field set to http://127.0.0.1:{PORT}/api/dashboard"
    )
    print(f"Thermal feed client listening on {THERMAL_HOST}:{THERMAL_PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down bridge...")
    finally:
        if THERMAL_FEED is not None:
            THERMAL_FEED.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
