# AGENTS.md

## Project overview

**PrusaSAM** is a Python + vanilla-JS monitoring stack for a Prusa MK4s printer via OctoPrint, with optional SenXor MI48 thermal imaging.

| Entry point | Purpose |
|---|---|
| `dashboard_bridge.py` | HTTP bridge + web dashboard host (`:8765`) |
| `rl_layer_logger.py` | Layer-triggered RL (speed/flow/fan) + IR + CNN — see [RL_RUNBOOK.md](RL_RUNBOOK.md) |
| `thermal_layer_logger.py` | Layer-triggered IR/MPC control (`fixed` / `mpc` / `mpc_temp`) — see [THERMAL_CONTROL_RUNBOOK.md](THERMAL_CONTROL_RUNBOOK.md) |
| `fixed_control.py` | Hold fixed speed/nozzle (no logging) until print end |
| `train_layer_defect_cnn.py` | Train webcam defect CNN (offline) |
| `prusa_dashboard.html` | Browser UI (demo mode or live via bridge) |
| `prusa_dashboard.py` | Tkinter desktop dashboard |
| `server.py` | SenXor MI48 → TCP thermal stream (`:5001`) |
| `octoprint_sensor_check.py` | OctoPrint status/position CLI |
| `M106-M107_Remove.py` | One-shot utility to strip fan G-code from a file |

## Local setup

Python **3.10+** is required. Dependencies are not pinned in-repo; install with pip (see VM update script).

Create a gitignored `api_key.py` with your OctoPrint API key before running live integrations:

```python
API_key = "your-octoprint-api-key"
```

OctoPrint URL is hardcoded to `http://cimpi-prusamk4s.local` in the Python modules.

## Run commands

| Service | Command | URL / port |
|---|---|---|
| Web bridge (primary dev path) | `python3 dashboard_bridge.py` | http://127.0.0.1:8765/ |
| Thermal stream server | `python3 server.py` | TCP `:5001` (needs SenXor USB) |
| Desktop dashboard | `python3 prusa_dashboard.py` | Tkinter (needs display) |
| OctoPrint probe | `python3 octoprint_sensor_check.py` | CLI |

Environment overrides for the bridge:

- `PRUSASAM_BRIDGE_HOST` (default `127.0.0.1`)
- `PRUSASAM_THERMAL_HOST` / `PRUSASAM_THERMAL_PORT` (default OctoPrint host, port `5001`)

## Validation

There is no pytest suite or linter config. Basic checks:

```bash
python3 -m py_compile dashboard_bridge.py rl_layer_logger.py thermal_layer_logger.py fixed_control.py layer_defect_cnn.py train_layer_defect_cnn.py octoprint_sensor_check.py prusa_dashboard.py server.py senxor/*.py
curl http://127.0.0.1:8765/api/health
python3 -m unittest test_rl_layer_logger.py test_layer_defect_cnn.py test_dashboard_bridge.py -v
```

## Cursor Cloud specific instructions

- **No `requirements.txt`** — pip packages are installed by the VM update script on startup.
- **`api_key.py` is gitignored** — create it locally for any module that imports `API_key` (`dashboard_bridge.py`, `octoprint_sensor_check.py`, `prusa_dashboard.py`). A placeholder key is enough to start the bridge; live OctoPrint data needs a real key and network access to the printer host.
- **Web dashboard without hardware**: run `dashboard_bridge.py` and open http://127.0.0.1:8765/. Without OctoPrint or the thermal server, `/api/dashboard` returns offline JSON but the UI still refreshes. Opening `prusa_dashboard.html` directly (`file://`) uses fully local browser demo data with animated thermal frames.
- **Live end-to-end** requires OctoPrint at `cimpi-prusamk4s.local`, optional `server.py` + SenXor MI48 USB on port 5001, and a valid API key.
- **Desktop app** (`prusa_dashboard.py`) needs a display server (X11/Wayland); prefer the web bridge in headless cloud VMs.
- **Long-running services**: use tmux (e.g. session `dashboard-bridge`) rather than background one-shot shells.
