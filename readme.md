# PrusaSAM Dashboard Bridge API

**RL + CNN workflow:** see [RL_RUNBOOK.md](RL_RUNBOOK.md) for what to run during prints, CNN training, and labeling.

This project includes a small HTTP bridge named `dashboard_bridge.py`.

The bridge lets another program:

1. Fetch the latest thermal matrix.
2. Fetch OctoPrint bed and hotend temperatures.
3. Send G-code commands to OctoPrint.
4. Read the same combined dashboard data used by the browser dashboard.

The bridge listens on:

```text
http://127.0.0.1:8765
```

## Quick start

### 1. Create an OctoPrint API key file

Create a file named `api_key.py` in this folder:

```python
API_key = "your-octoprint-api-key"
```

This file is intentionally ignored by git.

You can also provide the key with an environment variable:

```bash
export PRUSASAM_OCTOPRINT_API_KEY="your-octoprint-api-key"
```

### 2. Start the dashboard bridge

Run:

```bash
python3 dashboard_bridge.py
```

You should see output like:

```text
Serving PrusaSAM dashboard bridge on http://127.0.0.1:8765
Thermal feed client listening on cimpi-prusamk4s.local:5001
```

Leave this process running while your other Python program talks to the API.

### 3. Optional: start the thermal stream server

The thermal matrix comes from the SenXor thermal stream server on TCP port `5001`.

Run this only on the machine connected to the SenXor MI48 camera:

```bash
python3 server.py
```

If the thermal stream is not running, the API still works, but thermal fields will show an error or `null`.

## Base URL

All examples below use this base URL:

```text
http://127.0.0.1:8765
```

If another computer needs to reach the bridge, start the bridge with a reachable host:

```bash
PRUSASAM_BRIDGE_HOST=0.0.0.0 python3 dashboard_bridge.py
```

Then use the bridge computer's IP address instead of `127.0.0.1`.

Example:

```text
http://192.168.1.25:8765
```

## Webcam photo logging

The desktop dashboard webcam stream can save a photo every 15 seconds.

The output folder is set in `prusa_dashboard.py`:

```python
WEBCAM_PHOTO_LOG_PATH = Path(__file__).resolve().parent / "webcam_photo_log"
WEBCAM_PHOTO_LOG_INTERVAL_SECONDS = 15.0
```

By default, photos are saved in:

```text
webcam_photo_log/
```

Example file name:

```text
webcam_20260617_063000.jpg
```

To change where photos are saved, edit `WEBCAM_PHOTO_LOG_PATH`.

To change how often photos are saved, edit `WEBCAM_PHOTO_LOG_INTERVAL_SECONDS`.

The generated photo folder is ignored by git.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Check that the bridge is running. |
| `GET` | `/api/temperatures` | Fetch OctoPrint bed and hotend temperatures. |
| `GET` | `/api/thermal` | Fetch only the latest thermal camera frame. |
| `GET` | `/api/telemetry` | Fetch temperatures and thermal matrix together. |
| `GET` | `/api/dashboard` | Fetch the full browser dashboard payload. |
| `POST` | `/api/gcode` | Send G-code to OctoPrint. |

## Check bridge health

Use this to make sure the bridge is alive.

### curl

```bash
curl http://127.0.0.1:8765/api/health
```

### Python

```python
import json
from urllib.request import urlopen

with urlopen("http://127.0.0.1:8765/api/health", timeout=5) as response:
    data = json.loads(response.read().decode("utf-8"))

print(data)
```

Example response:

```json
{
  "ok": true,
  "time": "2026-06-09T21:04:12"
}
```

## Fetch temperatures

Use `/api/temperatures` when you only need bed and hotend temperature data.

### curl

```bash
curl http://127.0.0.1:8765/api/temperatures
```

### Python

```python
import json
from urllib.request import urlopen

BRIDGE_URL = "http://127.0.0.1:8765"

with urlopen(f"{BRIDGE_URL}/api/temperatures", timeout=10) as response:
    data = json.loads(response.read().decode("utf-8"))

bed = data["bed_temperature"]
hotend = data["temperatures"]["tool0"]

print("Bed actual:", bed["actual"])
print("Bed target:", bed["target"])
print("Hotend actual:", hotend["actual"])
print("Hotend target:", hotend["target"])
```

Example response:

```json
{
  "ok": true,
  "source": "octoprint",
  "updated_at": "2026-06-09 21:04:12",
  "bed_temperature": {
    "actual": 60.5,
    "target": 65.0,
    "offset": 0.0
  },
  "temperatures": {
    "bed": {
      "actual": 60.5,
      "target": 65.0,
      "offset": 0.0
    },
    "tool0": {
      "actual": 212.2,
      "target": 215.0,
      "offset": null
    }
  },
  "octoprint_url": "http://cimpi-prusamk4s.local"
}
```

## Fetch the thermal matrix

Use `/api/thermal` when you only need the thermal camera frame.

### curl

```bash
curl http://127.0.0.1:8765/api/thermal
```

### Python

```python
import json
from urllib.request import urlopen

BRIDGE_URL = "http://127.0.0.1:8765"

with urlopen(f"{BRIDGE_URL}/api/thermal", timeout=10) as response:
    data = json.loads(response.read().decode("utf-8"))

if not data["ok"]:
    print("No thermal frame:", data["error"])
else:
    thermal = data["thermal"]
    matrix = thermal["values"]

    print("Rows:", thermal["rows"])
    print("Columns:", thermal["cols"])
    print("Minimum temperature:", thermal["min"])
    print("Maximum temperature:", thermal["max"])
    print("First pixel:", matrix[0][0])
```

Example successful response:

```json
{
  "ok": true,
  "thermal": {
    "source": "thermal-pi",
    "connected": true,
    "updated_at": "2026-06-09 21:04:12",
    "error_text": "",
    "rows": 62,
    "cols": 80,
    "min": 24.0,
    "max": 31.5,
    "values": [
      [24.0, 24.1, 24.2]
    ]
  }
}
```

The real `values` field is a `62 x 80` matrix.

Each item is a temperature value from the thermal camera.

## Fetch temperatures and thermal matrix together

Use `/api/telemetry` when another program needs everything in one request.

This is usually the best endpoint for automation.

### curl

```bash
curl http://127.0.0.1:8765/api/telemetry
```

### Python

```python
import json
from urllib.request import urlopen

BRIDGE_URL = "http://127.0.0.1:8765"

with urlopen(f"{BRIDGE_URL}/api/telemetry", timeout=10) as response:
    data = json.loads(response.read().decode("utf-8"))

bed = data["bed_temperature"]
matrix = data["thermal_matrix"]

print("Bridge OK:", data["ok"])
print("Bed actual:", bed["actual"])
print("Bed target:", bed["target"])

if matrix is None:
    print("Thermal matrix unavailable:", data["errors"].get("thermal"))
else:
    print("Thermal rows:", data["thermal_rows"])
    print("Thermal columns:", data["thermal_cols"])
    print("Top-left thermal value:", matrix[0][0])
```

Example response:

```json
{
  "ok": true,
  "source": "python",
  "updated_at": "2026-06-09 21:04:12",
  "bed_temperature": {
    "actual": 60.5,
    "target": 65.0,
    "offset": 0.0
  },
  "temperatures": {
    "bed": {
      "actual": 60.5,
      "target": 65.0,
      "offset": 0.0
    },
    "tool0": {
      "actual": 212.2,
      "target": 215.0,
      "offset": null
    }
  },
  "thermal_matrix": [
    [24.0, 24.1, 24.2]
  ],
  "thermal_rows": 62,
  "thermal_cols": 80,
  "thermal": {
    "source": "thermal-pi",
    "connected": true,
    "updated_at": "2026-06-09 21:04:12",
    "error_text": "",
    "rows": 62,
    "cols": 80,
    "min": 24.0,
    "max": 31.5,
    "values": [
      [24.0, 24.1, 24.2]
    ]
  },
  "errors": {},
  "octoprint_url": "http://cimpi-prusamk4s.local"
}
```

If OctoPrint or the thermal stream is offline, the endpoint still returns JSON.

Example partial response:

```json
{
  "ok": false,
  "bed_temperature": {
    "actual": null,
    "target": null,
    "offset": null
  },
  "temperatures": {},
  "thermal_matrix": null,
  "thermal_rows": 62,
  "thermal_cols": 80,
  "errors": {
    "temperatures": "OctoPrint connection error",
    "thermal": "no_frame"
  }
}
```

## Send G-code

Use `POST /api/gcode` to send G-code through the bridge to OctoPrint.

The bridge forwards commands to OctoPrint's printer command API.

### Important safety notes

- G-code can move motors, heat the bed, heat the nozzle, start prints, or cancel operations.
- Only send commands you understand.
- Test with harmless commands first, such as `M105`.
- The bridge checks that commands are non-empty strings.
- The bridge does not maintain a whitelist of safe G-code commands.
- OctoPrint is still responsible for accepting or rejecting the command.

### Send one command with curl

```bash
curl -X POST http://127.0.0.1:8765/api/gcode \
  -H "Content-Type: application/json" \
  -d '{"command":"M105"}'
```

Example response:

```json
{
  "ok": true,
  "sent": ["M105"],
  "octoprint_status": 204
}
```

### Send multiple commands with curl

```bash
curl -X POST http://127.0.0.1:8765/api/gcode \
  -H "Content-Type: application/json" \
  -d '{"commands":["M105","M114"]}'
```

### Send one command with Python

```python
import json
from urllib.request import Request, urlopen

BRIDGE_URL = "http://127.0.0.1:8765"

payload = {"command": "M105"}

request = Request(
    f"{BRIDGE_URL}/api/gcode",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urlopen(request, timeout=10) as response:
    data = json.loads(response.read().decode("utf-8"))

print(data)
```

### Send multiple commands with Python

```python
import json
from urllib.request import Request, urlopen

BRIDGE_URL = "http://127.0.0.1:8765"

payload = {
    "commands": [
        "M105",
        "M114"
    ]
}

request = Request(
    f"{BRIDGE_URL}/api/gcode",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urlopen(request, timeout=10) as response:
    data = json.loads(response.read().decode("utf-8"))

print(data)
```

## G-code request format

Send exactly one of these fields:

### One command

```json
{
  "command": "M105"
}
```

### Multiple commands

```json
{
  "commands": [
    "M105",
    "M114"
  ]
}
```

Do not send both `command` and `commands` in the same request.

Do not send empty strings.

Do not send an empty list.

## Useful starter G-code commands

These commands are common for testing and status checks.

Always confirm that a command is safe for your printer before using it.

| Command | Meaning |
|---|---|
| `M105` | Report current temperatures. |
| `M114` | Report current position. |
| `G28` | Home axes. This moves the printer. |
| `M104 S0` | Turn off hotend heating. |
| `M140 S0` | Turn off bed heating. |
| `M18` | Disable steppers. |

## Error responses

### Empty command

Request:

```json
{
  "command": ""
}
```

Response:

```json
{
  "ok": false,
  "error": "G-code commands must not be empty."
}
```

### Empty command list

Request:

```json
{
  "commands": []
}
```

Response:

```json
{
  "ok": false,
  "error": "Provide at least one G-code command."
}
```

### OctoPrint unavailable

Response:

```json
{
  "ok": false,
  "error": "connection error details"
}
```

This means the bridge received your request, but could not complete the request against OctoPrint.

## Simple Python client class

You can copy this into another Python program.

```python
import json
from urllib.request import Request, urlopen


class PrusaSamBridgeClient:
    def __init__(self, base_url="http://127.0.0.1:8765"):
        self.base_url = base_url.rstrip("/")

    def get_json(self, path, timeout=10):
        with urlopen(f"{self.base_url}{path}", timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def post_json(self, path, payload, timeout=10):
        request = Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def health(self):
        return self.get_json("/api/health")

    def temperatures(self):
        return self.get_json("/api/temperatures")

    def thermal(self):
        return self.get_json("/api/thermal")

    def telemetry(self):
        return self.get_json("/api/telemetry")

    def send_gcode(self, command):
        return self.post_json("/api/gcode", {"command": command})

    def send_gcode_batch(self, commands):
        return self.post_json("/api/gcode", {"commands": commands})


client = PrusaSamBridgeClient()

print(client.health())
print(client.telemetry())
print(client.send_gcode("M105"))
```

## Recommended polling pattern

For monitoring programs, use `/api/telemetry`.

Example:

```python
import time

client = PrusaSamBridgeClient()

while True:
    telemetry = client.telemetry()
    bed = telemetry["bed_temperature"]
    matrix = telemetry["thermal_matrix"]

    print("Bed:", bed["actual"], "Target:", bed["target"])

    if matrix is not None:
        print("Thermal top-left:", matrix[0][0])

    time.sleep(1)
```

Avoid polling too fast. One request per second is a reasonable starting point for another monitoring program.

## Troubleshooting

### The bridge does not start

Check that `api_key.py` exists or set `PRUSASAM_OCTOPRINT_API_KEY`.

### Temperature fields are null

The bridge cannot reach OctoPrint, or OctoPrint returned no temperature data.

Check that the OctoPrint host is reachable:

```text
http://cimpi-prusamk4s.local
```

### Thermal matrix is null

The thermal stream is not connected or has not produced a frame yet.

Start the thermal stream server:

```bash
python3 server.py
```

Then call:

```bash
curl http://127.0.0.1:8765/api/thermal
```

### G-code returns an error

Check these items:

1. The JSON body must contain `command` or `commands`.
2. Commands must be non-empty strings.
3. OctoPrint must be reachable.
4. The API key must have permission to send printer commands.
5. The printer must be connected in OctoPrint.

## Summary

For most external Python programs:

1. Start the bridge with `python3 dashboard_bridge.py`.
2. Call `GET /api/telemetry` to read bed temperature and thermal matrix.
3. Call `POST /api/gcode` to send G-code.

The simplest useful Python calls are:

```python
telemetry = client.telemetry()
client.send_gcode("M105")
```
