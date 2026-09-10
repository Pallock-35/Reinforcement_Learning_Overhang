# RL Layer Logger — Runbook

Guide for running **PrusaSAM** layer-by-layer logging and reinforcement learning to study **maximum overhang angle** (from vertical) that a closed-loop system can print.

Each run records the tower angle, per-layer IR temperature, webcam photos, and optional CNN defect labels. Use **open-loop** mode to collect baseline data at fixed settings before enabling RL control.

For the HTTP bridge API alone, see [readme.md](readme.md).

---

## Architecture

```text
Pi (SenXor USB)          Laptop
  server.py :5001   →    dashboard_bridge.py :8765
                              ↑
                         rl_layer_logger.py
                              ↓
                         OctoPrint (MK4s)
                           M220 speed
                           M221 flow
                           M106/M107 fan
                           webcam snapshot
```

| Script | Role |
|--------|------|
| `server.py` | IR camera → TCP thermal stream |
| `dashboard_bridge.py` | Thermal + OctoPrint + G-code HTTP API |
| `rl_layer_logger.py` | Layer detection, RL agent (or open-loop logging), photos |
| `train_layer_defect_cnn.py` | Train webcam defect classifier (offline) |

---

## Research workflow (maximum angle)

1. **Baseline sweeps** — Print towers at increasing angles with **fixed** settings (170 mm/s, 100% flow, 100% fan) using `--open-loop`. Label layers after each print to find the maximum angle without sagging at baseline settings.
2. **Train CNN** — Train on labeled photos from baseline (and later RL) runs.
3. **Closed-loop RL** — Run without `--open-loop` at the same angles. The agent adjusts speed, flow, and fan each layer to push past the baseline limit.

Record the overhang angle for every print with `--print-angle` or the per-print prompt (0° = vertical, 90° = horizontal).

---

## One-time setup

1. **Create `api_key.py`** with your OctoPrint API key (gitignored).
2. **Install DisplayLayerProgress** on OctoPrint (layer detection).
3. **Strip fan from G-code** (`M106` / `M107`) so the logger owns cooling:

```bash
python M106-M107_Remove.py G-code/Overhang30-fan-front-170mm_s.gcode
python M106-M107_Remove.py G-code/input.gcode -o G-code/output-nofan.gcode
```
4. **Upload fan-stripped G-code** to OctoPrint before each job (use the G-code matched to the target overhang angle).
5. **Tune IR ROI** via env vars (`PRUSASAM_ROI_SKIP_TOP_MIN`, `PRUSASAM_ROI_N_ROWS`, etc.) using the thermal inspector at logger startup.

---

## Every print session

Start these **before** the print and leave them running:

**On the Pi** (SenXor connected):

```bash
python server.py
```

**On the laptop**:

```bash
python dashboard_bridge.py
```

Optional monitor: http://127.0.0.1:8765/

Health check:

```bash
curl http://127.0.0.1:8765/api/health
```

---

## Phase A — Baseline data at fixed settings (open-loop)

**When:** Sweeping overhang angles to find the maximum printable angle **without** closed-loop control, or collecting labeled photos before the CNN exists.

1. Start services above.
2. Upload G-code for the target angle and start the print on OctoPrint.
3. Run:

```bash
python rl_layer_logger.py --print-angle 30 --open-loop
```

`--open-loop` fixes every layer to **170 mm/s**, **100% flow**, **100% fan**; disables CNN inference and RL control. Omit `--print-angle` to enter the angle at the prompt before each print.

4. Label layers when prompted, e.g. `1-30:normal,31-50:defective`.
5. Outputs (one folder per print):

```text
print_runs/
  print_001/
    log.csv
    run_meta.json
    photos/
      layer_0001.jpg
      layer_0002.jpg
  print_002/
    log.csv
    run_meta.json
    photos/
      ...
```

`run_meta.json` stores `print_angle_deg`, `control_mode` (`open_loop` or `rl`), and base settings. Each `log.csv` row also includes `print_angle_deg` and `control_mode`.

Repeat at increasing angles until layers sag. Include **multiple filament colors** in labels if you print with different materials.

**Alternative (RL-style logging, no CNN):** `--no-cnn` without `--open-loop` still runs the Q-learning agent and varies speed/flow/fan. Prefer `--open-loop` for pure baseline angle sweeps.

---

## Phase B — Train the CNN (offline)

**When:** You have labeled photos and at least some **normal** and **defective** layers.

Create `layer_labels.json` (example — keys match print folder names under `print_runs/`):

```json
{
  "print_001": "1-30:normal,31-45:defective",
  "print 2": "1-20:normal,21-35:defective"
}
```

Keys like `"print_001"`, `"print 1"`, or `"1"` all refer to the same print folder (`print_001`). A legacy nested `"prints"` object is also supported.

Train (default **luminance** filter — see below):

```bash
python train_layer_defect_cnn.py --photo-root print_runs --layer-labels-file layer_labels.json
```

Or from logs that already have `visual_defect` filled in:

```bash
python train_layer_defect_cnn.py --photo-root print_runs --logs "print_runs/print_*/log.csv"
```

**Output:** `models/layer_defect_cnn.pt`

Retrain after adding more labeled prints.

---

## Phase C — Closed-loop RL with CNN

**When:** `models/layer_defect_cnn.pt` exists and you want to test whether RL can print steeper overhangs than the open-loop baseline.

1. Start `server.py` + `dashboard_bridge.py`.
2. Upload G-code for the target angle and start the print on OctoPrint.
3. Run:

```bash
python rl_layer_logger.py --print-angle 35
```

The agent each layer:

1. Captures a webcam photo → CNN classifies normal / defective / unknown.
2. Reads **deposit temperature** from the Senxor IR ROI (`deposit_temp_c`).
3. Picks one of nine actions: nudge **speed** (M220), **flow** (M221), or **fan** (M106/M107).
4. Updates the Q-table in `q_tables/` (shape changed from older speed-only tables — old files are skipped automatically).

Compare results at the same `print_angle_deg` against open-loop runs in `print_runs/`.

---

## CNN image filtering (filament colors)

The CNN does **not** use raw RGB by default. Photos are converted to **contrast-normalized luminance** (grayscale brightness) before training and inference. That reduces sensitivity to red vs blue filament while keeping layer shape and sagging cues.

| Mode | Command | Use when |
|------|---------|----------|
| `luminance` (default) | `train_layer_defect_cnn.py` or env `PRUSASAM_CNN_INPUT_MODE=luminance` | Multi-color filament, recommended |
| `rgb` | `--input-mode rgb` | Debugging only; retrain required |

The mode is stored inside `models/layer_defect_cnn.pt`. Inference uses the same filter automatically.

**Best practice:** Still include several filament colors in your **labeled training set** — filtering helps, but diverse examples improve robustness.

---

## Labeling and CLI cheat sheet

| Task | Command |
|------|---------|
| Baseline angle sweep (fixed settings) | `python rl_layer_logger.py --print-angle 30 --open-loop` |
| Closed-loop RL at an angle | `python rl_layer_logger.py --print-angle 35` |
| Prompt for angle each print | Omit `--print-angle` (defaults to last angle used) |
| Label before print | Run `rl_layer_logger.py` and answer the layer-label prompt |
| Label from JSON | `--layer-labels-file layer_labels.json` |
| Label all prints same ranges | `--layer-labels "1-30:normal,31-50:defective"` |
| Fix an old log | `python rl_layer_logger.py --apply-labels-to print_runs/print_001/log.csv --layer-labels "..."` |
| Run CNN on old log | `python rl_layer_logger.py --apply-cnn-to print_runs/print_001/log.csv` |
| Skip photo capture | `--no-layer-photos` |
| Skip CNN (RL still active) | `--no-cnn` |
| Skip label prompts | `--no-label-prompt` |

Label syntax: `1-30:normal,31-50:defective,51:normal` (aliases: `ok`, `good`, `sagging`, `bad`).

---

## Controls and rewards (Section 1 of `rl_layer_logger.py`)

### Open-loop baseline (`--open-loop`)

| Parameter | Value |
|-----------|-------|
| Print speed | 170 mm/s (M220 100%) |
| Flow | 100% (M221) |
| Fan | 100% (M106) |

No Q-table updates. CNN is disabled automatically.

### RL mode (default without `--open-loop`)

| Parameter | G-code | Levels |
|-----------|--------|--------|
| Print speed | M220 | 65–160 mm/s (discrete); base reference 170 mm/s |
| Flow | M221 | 85–110% |
| Fan | M106 / M107 | 0, 25, 50, 75, 100% |

IR target band: `BAND_LO`–`BAND_HI` °C (default 60–90) on deposit ROI temperature — used in RL **reward** and logging (`in_window` column), not the primary angle metric.

CNN affects RL via **state** (visual bin) and **reward** (penalty for defective, bonus for normal + in-band temp).

---

## Typical day

```text
1. server.py                                    (Pi)
2. dashboard_bridge.py                          (laptop)
3. Upload fan-stripped G-code → print on OctoPrint
4. rl_layer_logger.py --print-angle 30 --open-loop   (baseline at 30°)
5. Repeat step 4 at 35°, 40°, … until sagging; label each run
6. train_layer_defect_cnn.py                  (once enough labels)
7. rl_layer_logger.py --print-angle 35          (closed-loop RL at same angles)
```

---

## Outputs (gitignored)

| Path | Contents |
|------|----------|
| `print_runs/` | Per-print folders (`log.csv`, `run_meta.json`, `photos/`) |
| `models/` | Trained CNN weights |
| `q_tables/` | Q-learning tables (RL mode only) |

---

## Other scripts

| Script | When |
|--------|------|
| `octoprint_sensor_check.py` | Debug OctoPrint connectivity |
| `M106-M107_Remove.py` | One-shot: strip fan lines from a G-code file |
| `prusa_dashboard.py` / browser at `:8765` | Monitoring only |

---

## Python layout (after cleanup)

| File | Role |
|------|------|
| `server.py` | IR thermal stream (Pi) |
| `dashboard_bridge.py` | HTTP bridge + web dashboard |
| `rl_layer_logger.py` | Main RL / open-loop logging + CNN inference |
| `layer_defect_cnn.py` | CNN model, training, inference |
| `train_layer_defect_cnn.py` | CNN training CLI |
| `thermal_roi.py` | Senxor deposit ROI summarization |
| `octoprint_sensor_check.py` | OctoPrint debug CLI |
| `prusa_dashboard.py` | Optional Tkinter desktop UI |
| `M106-M107_Remove.py` | Strip fan G-code from a file |
| `senxor/` | SenXor MI48 driver (used by `server.py`) |
| `test_*.py` | Unit tests |

Legacy scripts removed: `main_control 1.py`, `data_collector.py`, `layer_data_collector.py` (all superseded by `rl_layer_logger.py`).

---

```bash
python -m unittest test_rl_layer_logger.py test_layer_defect_cnn.py -v
```
