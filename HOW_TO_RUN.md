# How to Run PrusaSAM (MPC Temp Print)

Step-by-step guide to start a live overhang print with the dashboard bridge and `thermal_layer_logger.py`.

---

## 1. Open OctoPrint

1. Open **Google Chrome**.
2. In the address bar, go to:

```text
http://10.105.77.241
```

This is the OctoPrint IP for the Prusa MK4s.

3. Log in with:

| Field | Value |
|-------|-------|
| Username | `cimpi` |
| Password | `etrl236` |

4. Start the overhang coupon you want to print. Overhang G-code files are in the **overhangs** folder on OctoPrint.

---

## 2. Start the dashboard bridge

In a terminal, from the PrusaSAM project folder, copy and paste:

```bash
python dashboard_bridge.py
```

Leave this terminal open. The bridge serves the dashboard and APIs at `http://127.0.0.1:8765`.

---

## 3. Open the dashboard in Chrome

1. With the bridge running, open the dashboard in Google Chrome by pasting this into the address bar:

```text
http://127.0.0.1:8765/
```

You can also open `prusa_dashboard.html` from the project folder in Chrome, but prefer the URL above so the page is served by the bridge.

2. Confirm the dashboard is **connected to the bridge** (live mode), not offline demo. If it shows demo / offline, make sure `dashboard_bridge.py` is still running and refresh the page.

---

## 4. Start the thermal layer logger (new terminal)

Open a **new terminal**, go to the PrusaSAM project folder, and copy and paste:

```bash
python thermal_layer_logger.py --control-mode mpc_temp --print-speed 100 --nozzle-temp 210 --print-angle 70 --temperature-target 46.6 --nozzle-min 190 --nozzle-max 230 --max-nozzle-step 3 --prints 1 --model-params models/thermal_fit.json
```

**Important:** set `--print-angle` to match the overhang coupon you are printing (for example `60`, `65`, `70`, `75`, or `80`). The example above uses `70`.

---

## 5. Arm the logger at the start of the print

When the printer begins the first layer, or while it prints the purge line at the start of the print, **press Enter** in the thermal logger terminal.

Do this at the start of every print so logging and control begin on time.

---

## Checklist

- [ ] OctoPrint open at `http://10.105.77.241` (login `cimpi` / `etrl236`)
- [ ] Overhang job selected from the **overhangs** folder and started
- [ ] `python dashboard_bridge.py` running
- [ ] Dashboard open at `http://127.0.0.1:8765/` and connected (not demo)
- [ ] Logger running in a second terminal with `--print-angle` matching the coupon
- [ ] Enter pressed during the first layer / purge line
