# Thermal Control Runbook

Guide for **IR-informed MPC overhang control** in PrusaSAM: layerwise `roi_mean` feedback, per-layer thermal-dose proxy `D_k`, and three live control modes (fixed / mpc / mpc_temp).

Paper plan: [IR_informed_MPC_overhang_outline_MathJax.md](IR_informed_MPC_overhang_outline_MathJax.md).  
For the legacy Q-learning logger, see [RL_RUNBOOK.md](RL_RUNBOOK.md).

---

## Architecture

```text
Pi (SenXor USB)          Laptop
  server.py :5001   →    dashboard_bridge.py :8765
                              ↑
                         thermal_layer_logger.py
                              ↓
                         OctoPrint (MK4s)
                           M220 speed (base 170 mm/s)
                           M221 flow (fixed 100%)
                           M106 S255 fan (always 100%)
                           M104 nozzle setpoint
```

| Module | Role |
|--------|------|
| `thermal_layer_logger.py` | Live layer loop, logging, `fixed` / `mpc` / `mpc_temp` |
| `fixed_control.py` | Same fixed setpoints as `--control-mode fixed`, no CSV/stream logs |
| `thermal_model.py` | Analytical thermal step + parameter fit (`u = [v, T_nozzle]`) |
| `mpc_controller.py` | SLSQP MPC solver + offline simulators |
| `thermal_dose.py` | Per-layer `D_k` accumulation from `T_k` |
| `thermal_targets.py` | `T*(α)`, `D*(α)`, `D*±ΔD`, bond/sag limits |
| `printer_control.py` | Bridge IR read + G-code actuation |
| `baseline_controllers.py` | Fixed `[v, T_nozzle]` baseline |
| `calibrate_thermal_model.py` | Fit θ; report RMSE, R², short-horizon `e_N` |
| `analyze_thermal_runs.py` | `E_T`, `E_D`, `V_D`, `TV_u`, `t_print` |

**Primary feedback:** `T_k = roi_mean` from the deposit ROI ([thermal_roi.py](thermal_roi.py)).

**Dose semantics:** `D_k` is the integral of `max(T - T_g, 0)` over the completed layer only. The logger resets the dose tracker after each layer decision so targets `D*(α) ± ΔD` stay on a per-layer scale.

**Control vector:** `u = [v_mms, T_nozzle_C]`. Fan is locked at 100%; dwell is always 0.

---

## Prerequisites

1. `api_key.py` with OctoPrint API key (or `PRUSASAM_OCTOPRINT_API_KEY`).
2. `dashboard_bridge.py` running at `http://127.0.0.1:8765`.
3. `server.py` on the Pi for live IR (optional for `--dry-run`).
4. DisplayLayerProgress plugin on OctoPrint.
5. Strip fan G-code from tower files if the logger owns cooling (`M106-M107_Remove.py`).

---

## Experiment workflow

```text
fixed: same V and T_nozzle, sweep --print-angle
        → find max printable angle without MPC
        → mpc or mpc_temp at steeper angles
```

| Mode | CLI | Actuators |
|------|-----|-----------|
| Baseline | `--control-mode fixed --print-speed V --nozzle-temp T` | Hold `V` and `T` for whole print; fan 100% |
| MPC speed+temp | `--control-mode mpc --print-speed V --nozzle-temp T` | MPC varies print speed and nozzle each layer; starts near `V`/`T` |
| MPC temp only | `--control-mode mpc_temp --print-speed V --nozzle-temp T` | Speed fixed at `V`; MPC varies nozzle only toward a fixed ROI-mean target |

Flow is held at **100%** (M221). Fan is always **M106 S255**. No `G4` dwell.

MPC prediction uses a layer-scale path length (default 300 mm) so `Δt = L/v` matches live layer decisions. Within each predicted layer the analytical segment model is integrated with **12 substeps** to approximate continuous deposition. Soft constraints include the dose window and bond/sag temperature limits.

Suggested MPC bounds: speed **50–170 mm/s**, nozzle **190–230 °C**. Fixed `--print-speed` can be any M220-mappable value.

### `mpc_temp` commissioning options

Temperature-only commissioning disables dose/window soft costs, pins speed, and tracks a fixed ROI-mean target (`T_target_override_c`):

| Flag | Default | Meaning |
|------|---------|---------|
| `--temperature-target` | 46.6 °C | Fixed ROI-mean target |
| `--nozzle-min` / `--nozzle-max` | 200 / 220 °C | Absolute nozzle bounds |
| `--max-nozzle-step` | 3 °C | Hard per-layer nozzle rate limit vs last applied command |

Live `mpc` / `mpc_temp` runs require a validated `--model-params` JSON (blocked if missing; `--dry-run` warns and falls back to defaults). Each `mpc_temp` print folder also writes `controller_config.json` with the active bounds/weights/model snapshot.

---

## Example commands

```bash
# Connection test (via bridge)
python thermal_layer_logger.py --control-mode fixed --dry-run

# Hold fixed speed/nozzle only (no logging)
python fixed_control.py --print-speed 50 --nozzle-temp 215

# Angle sweep without MPC (same speed/temp, different --print-angle each job)
python thermal_layer_logger.py --control-mode fixed --print-speed 50 --nozzle-temp 215 --print-angle 60

# Push past baseline with MPC (speed + nozzle)
python thermal_layer_logger.py --control-mode mpc --print-speed 50 --nozzle-temp 215 --print-angle 70 --model-params models/thermal_fit.json

# Push with nozzle only (speed held at 50 mm/s; commissioning defaults)
python thermal_layer_logger.py --control-mode mpc_temp --print-speed 100 --nozzle-temp 210 --print-angle 70 --model-params models/thermal_fit.json

# Optional commissioning overrides
python thermal_layer_logger.py --control-mode mpc_temp --print-speed 100 --nozzle-temp 210 --print-angle 70 \
  --temperature-target 46.6 --nozzle-min 200 --nozzle-max 220 --max-nozzle-step 3 \
  --model-params models/thermal_fit.json

# Fit model from existing logs (RMSE, R², e_N)
python calibrate_thermal_model.py print_runs/print_001_fixed_a60/log.csv --output-dir print_runs/calibration_fit

# Analyze experiment matrix (MAE metrics + mode×angle aggregate)
python analyze_thermal_runs.py print_runs --json-out print_runs/metrics.json

# Offline controller comparison
python ir_informed_mpc_overhang.py
```

---

## Log schema (`print_runs/print_NNN_<mode>_a<angle>/log.csv`)

Extended columns include: `T_k`, `D_k`, `D_min`, `D_max`, `D_star`, `T_star`, `dT_dt`, `nozzle_setpoint_c`, `fan_percent` (always 100), `dwell_s` (always 0), `mpc_cost`, `mpc_success`, `control_source`, `stream_samples`.

Between decisions, `stream.csv` records 1 Hz IR samples used to integrate the per-layer dose.

---

## Experiment matrix

Minimum: **3 replicates × 3 angles × {fixed, mpc, mpc_temp}**.

Use pre-sliced towers in `G-code/` (Overhang60–80). Record angle with `--print-angle`.

Compare runs:

```bash
python analyze_thermal_runs.py print_runs
```

Metrics follow outline §9:

| Symbol | Meaning |
|--------|---------|
| `E_T` | Mean absolute temperature tracking error |
| `E_D` | Mean absolute dose tracking error |
| `V_D` | Thermal-window violation rate |
| `TV_u` | Control total variation |
| `t_print` | Sum of layer intervals |

---

## Offline validation

```bash
python -m py_compile thermal_model.py mpc_controller.py thermal_layer_logger.py printer_control.py baseline_controllers.py
python -m unittest test_thermal_model.py test_mpc_controller.py test_printer_control.py test_print_run_naming.py test_thermal_dose.py test_thermal_targets.py -v
python ir_informed_mpc_overhang.py
```

---

## Dependencies

| Package | Required for |
|---------|----------------|
| `numpy`, `scipy` | MPC, model, calibration |
| `requests` | Bridge HTTP |
| `gymnasium`, `stable-baselines3` | Optional offline SAC training (`train_sac_thermal.py`) |
