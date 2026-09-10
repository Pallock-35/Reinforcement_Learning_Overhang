from __future__ import annotations

import base64
import json
import os
import random
import re
import sys
import ssl
import socket
import struct
import time
import uuid
from typing import Any
import threading
from typing import Optional
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from api_key import API_key


OCTOPRINT_URL = "http://cimpi-prusamk4s.local"
POSITION_RE = re.compile(
    r"(?:^|\s)X:\s*(?P<x>[-+]?\d+(?:\.\d+)?)\s+Y:\s*(?P<y>[-+]?\d+(?:\.\d+)?)\s+Z:\s*(?P<z>[-+]?\d+(?:\.\d+)?)"
)


def fetch_json(url: str, api_key: str) -> dict[str, Any]:
    request = Request(url, headers={"X-Api-Key": api_key})

    with urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_printer_status(base_url: str, api_key: str) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/printer"
    return fetch_json(url, api_key)


def fetch_passive_login(base_url: str, api_key: str) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/login"
    body = json.dumps({"passive": True}).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={
            "X-Api-Key": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def build_sockjs_websocket_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    host = parsed.hostname or "localhost"
    port = parsed.port

    if port is None:
        netloc = host
    else:
        netloc = f"{host}:{port}"

    prefix = parsed.path.rstrip("/")
    sockjs_path = f"{prefix}/sockjs" if prefix else "/sockjs"
    session_id = uuid.uuid4().hex
    server_id = random.randrange(0, 1000)

    return f"{scheme}://{netloc}{sockjs_path}/{server_id:03d}/{session_id}/websocket"


def _create_websocket_connection(url: str):
    parsed = urlparse(url)
    secure = parsed.scheme == "wss"
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if secure else 80)
    path = parsed.path or "/"

    raw_socket = socket.create_connection((host, port), timeout=10)
    if secure:
        context = ssl.create_default_context()
        raw_socket = context.wrap_socket(raw_socket, server_hostname=host)

    key = base64.b64encode(os.urandom(16)).decode("ascii")
    host_header = host if (port in (80, 443)) else f"{host}:{port}"
    request_lines = [
        f"GET {path} HTTP/1.1",
        f"Host: {host_header}",
        "Upgrade: websocket",
        "Connection: Upgrade",
        f"Sec-WebSocket-Key: {key}",
        "Sec-WebSocket-Version: 13",
        "\r\n",
    ]
    raw_socket.sendall("\r\n".join(request_lines).encode("ascii"))

    response = _read_http_headers(raw_socket)
    if not response.startswith("HTTP/1.1 101"):
        raise RuntimeError(f"WebSocket handshake failed: {response.splitlines()[0] if response else 'no response'}")

    accept_key = None
    for line in response.split("\r\n"):
        if line.lower().startswith("sec-websocket-accept:"):
            accept_key = line.split(":", 1)[1].strip()
            break

    expected_accept = base64.b64encode(
        __import__("hashlib").sha1(
            (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")
        ).digest()
    ).decode("ascii")
    if accept_key != expected_accept:
        raise RuntimeError("WebSocket handshake validation failed")

    raw_socket.settimeout(1.0)
    return raw_socket


def _read_http_headers(sock: socket.socket) -> str:
    buffer = bytearray()
    while b"\r\n\r\n" not in buffer:
        chunk = sock.recv(4096)
        if not chunk:
            break
        buffer.extend(chunk)
    return buffer.decode("latin-1", errors="replace")


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    buffer = bytearray()
    while len(buffer) < size:
        chunk = sock.recv(size - len(buffer))
        if not chunk:
            raise EOFError("Socket closed while receiving data")
        buffer.extend(chunk)
    return bytes(buffer)


def _send_websocket_text(sock: socket.socket, text: str) -> None:
    payload = text.encode("utf-8")
    mask = os.urandom(4)
    header = bytearray([0x81])
    length = len(payload)

    if length < 126:
        header.append(0x80 | length)
    elif length < 65536:
        header.append(0x80 | 126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(0x80 | 127)
        header.extend(struct.pack("!Q", length))

    masked_payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    sock.sendall(bytes(header) + mask + masked_payload)


def _send_sockjs_message(sock: socket.socket, data: dict[str, Any]) -> None:
    wrapped = '["' + json.dumps(data).replace('"', '\\"') + '"]'
    _send_websocket_text(sock, wrapped)


def _recv_websocket_frame(sock: socket.socket) -> tuple[int, bytes] | None:
    first_two = sock.recv(2)
    if not first_two:
        return None
    while len(first_two) < 2:
        more = sock.recv(2 - len(first_two))
        if not more:
            return None
        first_two += more

    first_byte, second_byte = first_two
    opcode = first_byte & 0x0F
    masked = bool(second_byte & 0x80)
    payload_length = second_byte & 0x7F

    if payload_length == 126:
        payload_length = struct.unpack("!H", _recv_exact(sock, 2))[0]
    elif payload_length == 127:
        payload_length = struct.unpack("!Q", _recv_exact(sock, 8))[0]

    mask_key = _recv_exact(sock, 4) if masked else b""
    payload = _recv_exact(sock, payload_length) if payload_length else b""
    if masked:
        payload = bytes(
            byte ^ mask_key[index % 4] for index, byte in enumerate(payload)
        )

    return opcode, payload


def _recv_sockjs_messages(sock: socket.socket, timeout: float) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    messages: list[dict[str, Any]] = []

    while time.monotonic() < deadline:
        remaining = max(deadline - time.monotonic(), 0.1)
        sock.settimeout(remaining)
        try:
            frame = _recv_websocket_frame(sock)
        except socket.timeout:
            break

        if frame is None:
            break

        opcode, payload = frame
        if opcode == 0x8:
            break
        if opcode == 0x9:
            _send_websocket_pong(sock, payload)
            continue
        if opcode != 0x1:
            continue

        data = payload.decode("utf-8", errors="replace")
        if data == "o" or data == "h":
            continue
        if data.startswith("c"):
            raise RuntimeError(f"SockJS closed connection: {data}")
        if not data.startswith("a"):
            continue

        try:
            entries = json.loads(data[1:])
        except json.JSONDecodeError:
            continue

        for entry in entries:
            if isinstance(entry, dict):
                messages.append(entry)
                continue
            try:
                messages.append(json.loads(entry))
            except json.JSONDecodeError:
                continue

    return messages


def _send_websocket_pong(sock: socket.socket, payload: bytes) -> None:
    mask = os.urandom(4)
    header = bytearray([0x8A])
    length = len(payload)

    if length < 126:
        header.append(0x80 | length)
    elif length < 65536:
        header.append(0x80 | 126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(0x80 | 127)
        header.extend(struct.pack("!Q", length))

    masked_payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    sock.sendall(bytes(header) + mask + masked_payload)


def post_printer_command(base_url: str, api_key: str, command: str) -> None:
    url = f"{base_url.rstrip('/')}/api/printer/command"
    body = json.dumps({"command": command}).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={
            "X-Api-Key": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urlopen(request, timeout=10):
        return None


def extract_position_from_text(text: str) -> dict[str, float] | None:
    match = POSITION_RE.search(text)
    if not match:
        return None

    return {
        "x": float(match.group("x")),
        "y": float(match.group("y")),
        "z": float(match.group("z")),
    }


class PushListener:
    """Background listener that keeps a SockJS push connection open and
    maintains the latest observed position.

    Use `start()` to begin and `stop()` to end. Call `get_position()` to
    retrieve the most recent position dict (x,y,z) where values may be
    NaN when not available.
    """

    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self._lock = threading.Lock()
        self._position: dict[str, float] = {"x": float("nan"), "y": float("nan"), "z": float("nan")}
        self._last_update: float | None = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def get_position(self) -> dict[str, float]:
        with self._lock:
            return dict(self._position)

    def get_last_update_age(self) -> float | None:
        """Return seconds since last position update, or None if never updated."""
        with self._lock:
            if self._last_update is None:
                return None
            return time.monotonic() - self._last_update

    def _set_position(self, pos: dict[str, float]) -> None:
        with self._lock:
            self._position.update(pos)
            self._last_update = time.monotonic()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                login = fetch_passive_login(self.base_url, self.api_key)
                user_id = login.get("name")
                session_id = login.get("session")
                if not user_id or not session_id:
                    raise RuntimeError("login did not return user/session")

                sock_url = build_sockjs_websocket_url(self.base_url)
                sock = _create_websocket_connection(sock_url)
                try:
                    _send_sockjs_message(sock, {"subscribe": {"state": {"logs": True, "messages": True}}})
                    _send_sockjs_message(sock, {"auth": f"{user_id}:{session_id}"})

                    # Read continuously; short timeouts let us detect stop quickly.
                    while not self._stop.is_set():
                        messages = _recv_sockjs_messages(sock, timeout=0.5)
                        if not messages:
                            continue
                        for message in messages:
                            payload = message.get("current") or message.get("history") or {}
                            if not isinstance(payload, dict):
                                continue

                            # Prefer explicit currentZ value when available
                            if isinstance(payload.get("currentZ"), (int, float)):
                                self._set_position({"z": float(payload["currentZ"])})

                            for field in ("logs", "messages"):
                                for line in payload.get(field, []) or []:
                                    if not isinstance(line, str):
                                        continue
                                    pos = extract_position_from_text(line)
                                    if pos is not None:
                                        self._set_position(pos)
                finally:
                    try:
                        sock.close()
                    except Exception:
                        pass
            except Exception:
                # On any error, wait briefly then retry connection.
                if self._stop.wait(0.5):
                    break
                continue



def fetch_position_via_m114(base_url: str, api_key: str, timeout: float = 2.5) -> dict[str, float] | None:
    login = fetch_passive_login(base_url, api_key)
    user_id = login.get("name")
    session_id = login.get("session")
    if not user_id or not session_id:
        raise RuntimeError("OctoPrint passive login did not return a user/session pair")

    socket_url = build_sockjs_websocket_url(base_url)
    sock = _create_websocket_connection(socket_url)
    current_z: float | None = None
    sample_lines: list[str] = []
    try:
        _send_sockjs_message(sock, {"subscribe": {"state": {"logs": True, "messages": True}}})
        _send_sockjs_message(sock, {"auth": f"{user_id}:{session_id}"})

        # Give the push stream a short moment to provide a position before we
        # fall back to the slower M114 request.
        messages = _recv_sockjs_messages(sock, timeout=0.4)
        for message in messages:
            payload = message.get("current") or message.get("history") or {}
            if not isinstance(payload, dict):
                continue

            if current_z is None and isinstance(payload.get("currentZ"), (int, float)):
                current_z = float(payload["currentZ"])

            for field in ("logs", "messages"):
                for line in payload.get(field, []) or []:
                    if not isinstance(line, str):
                        continue
                    if len(sample_lines) < 12:
                        sample_lines.append(line)
                    position = extract_position_from_text(line)
                    if position is not None:
                        return position

        post_printer_command(base_url, api_key, "M114")

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            messages = _recv_sockjs_messages(sock, timeout=max(deadline - time.monotonic(), 0.1))
            for message in messages:
                payload = message.get("current") or message.get("history") or {}
                if not isinstance(payload, dict):
                    continue

                if current_z is None and isinstance(payload.get("currentZ"), (int, float)):
                    current_z = float(payload["currentZ"])

                for field in ("logs", "messages"):
                    for line in payload.get(field, []) or []:
                        if not isinstance(line, str):
                            continue
                        if len(sample_lines) < 12:
                            sample_lines.append(line)
                        position = extract_position_from_text(line)
                        if position is not None:
                            return position
    finally:
        try:
            sock.close()
        except Exception:
            pass

    if current_z is not None:
        return {"x": float("nan"), "y": float("nan"), "z": current_z}

    if sample_lines:
        print("OctoPrint position debug: received serial lines:", file=sys.stderr)
        for line in sample_lines:
            print(f"  {line}", file=sys.stderr)

    return None


def summarize_sensor_data(payload: dict[str, Any]) -> str:
    temperature = payload.get("temperature", {})
    state = payload.get("state", {})

    summary_lines = ["OctoPrint printer status:"]
    if isinstance(state, dict) and state:
        summary_lines.append(f"- State: {state.get('text', 'unknown')}")

    if isinstance(temperature, dict) and temperature:
        summary_lines.append("- Temperature sensors:")
        for sensor_name, sensor_data in temperature.items():
            if not isinstance(sensor_data, dict):
                continue

            actual = sensor_data.get("actual")
            target = sensor_data.get("target")
            offset = sensor_data.get("offset")
            summary_lines.append(
                f"  - {sensor_name}: actual={actual}, target={target}, offset={offset}"
            )
    else:
        summary_lines.append("- No temperature sensor data returned.")

    return "\n".join(summary_lines)


def summarize_position(position: dict[str, float] | None) -> str:
    if position is None:
        return "- Position: unavailable"

    if any(value != value for value in position.values()):
        return f"- Position: z={position['z']:.3f} (XY unavailable)"

    return (
        "- Position: "
        f"x={position['x']:.3f}, y={position['y']:.3f}, z={position['z']:.3f}"
    )


def main() -> int:
    try:
        payload = fetch_printer_status(OCTOPRINT_URL, API_key)
        position = fetch_position_via_m114(OCTOPRINT_URL, API_key)
    except HTTPError as error:
        print(f"HTTP error while contacting OctoPrint: {error.code} {error.reason}", file=sys.stderr)
        return 1
    except URLError as error:
        print(f"Network error while contacting OctoPrint: {error.reason}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"Socket error while contacting OctoPrint: {error}", file=sys.stderr)
        return 1
    except RuntimeError as error:
        print(f"OctoPrint position query failed: {error}", file=sys.stderr)
        position = None

    print(summarize_sensor_data(payload))
    print(summarize_position(position))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())