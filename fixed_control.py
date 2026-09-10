#!/usr/bin/env python3
"""
fixed_control.py
================
Hold fixed print speed and nozzle temperature for a whole print — no logging.

Same setpoints as thermal_layer_logger.py --control-mode fixed:
  u = [v_mms, T_nozzle_C]
  flow 100% (M221), fan 100% (M106 S255), dwell 0

Applies controls at print start and again on each layer change so OctoPrint
keeps the configured values. Exits when the print finishes or on Ctrl+C.

Depends on dashboard_bridge.py for G-code actuation and print-state polling.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime

from baseline_controllers import (
    DEFAULT_NOZZLE_TEMP_C,
    DEFAULT_PRINT_SPEED_MMS,
    fixed_control,
)
from printer_control import (
    BASE_FLOW_PERCENT,
    apply_control_vector,
    clear_thermal_control_status,
    publish_thermal_control_status,
)
from rl_layer_logger import (
    BRIDGE_URL,
    DEFAULT_OCTOPRINT_URL,
    LayerChangeWatcher,
    SPEED_SETTLE_TIME,
    bridge_health,
    load_octoprint_api_key,
    read_printer_temps,
)

CONTROL_MODE = "fixed"


def test_connections() -> bool:
    """Check bridge + OctoPrint (thermal camera not required)."""
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

    print("\n  2. OctoPrint via bridge dashboard...")
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

    return all_ok


def run_fixed_control(
    octoprint_url: str,
    api_key: str,
    *,
    print_speed_mms: float,
    nozzle_temp_c: float,
    dry_run: bool = False,
) -> None:
    u = fixed_control(print_speed_mms, nozzle_temp_c)
    speed_mms = float(u[0])
    nozzle_setpoint_c = float(u[1])

    watcher = LayerChangeWatcher(octoprint_url, api_key)
    watcher.start()

    print(f"\n{'='*72}")
    print(f"  FIXED CONTROL  |  {datetime.now().strftime('%H:%M:%S')}")
    print(
        f"  Hold: speed={speed_mms:.0f} mm/s  nozzle={nozzle_setpoint_c:.0f}C  "
        f"flow=100%  fan=100%"
    )
    print(f"  Logging: off")
    print(f"{'='*72}")

    layers_held = 0

    try:
        if not dry_run:
            commands = apply_control_vector(u, flow_percent=BASE_FLOW_PERCENT)
            print(f"  Applied: {', '.join(commands)}")
            publish_thermal_control_status(
                control_mode=CONTROL_MODE,
                speed_mms=speed_mms,
                nozzle_temp_c=nozzle_setpoint_c,
                print_number=0,
                layer=None,
            )

        watcher.wait_for_print_start()

        while True:
            layer = watcher.wait_for_layer_complete()
            if layer is None:
                print("\n  Print finished — releasing fixed control.")
                break

            layers_held += 1
            if not dry_run:
                apply_control_vector(u, flow_percent=BASE_FLOW_PERCENT)
                publish_thermal_control_status(
                    control_mode=CONTROL_MODE,
                    speed_mms=speed_mms,
                    nozzle_temp_c=nozzle_setpoint_c,
                    print_number=0,
                    layer=layer,
                )
            time.sleep(SPEED_SETTLE_TIME)
            print(
                f"  Layer {layer:>4}  — held "
                f"{speed_mms:.0f} mm/s / {nozzle_setpoint_c:.0f}C"
            )

    except KeyboardInterrupt:
        print("\n  Interrupted — releasing fixed control.")
    finally:
        watcher.stop()
        if not dry_run:
            clear_thermal_control_status()

    print(f"\n  Done — held setpoints across {layers_held} layer(s).")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Hold fixed print speed and nozzle temp until print end "
            "(no logging; same actuators as --control-mode fixed)"
        )
    )
    parser.add_argument(
        "--print-speed",
        type=float,
        default=DEFAULT_PRINT_SPEED_MMS,
        metavar="MMS",
        help=f"Print speed mm/s (default {DEFAULT_PRINT_SPEED_MMS:g})",
    )
    parser.add_argument(
        "--nozzle-temp",
        type=float,
        default=DEFAULT_NOZZLE_TEMP_C,
        metavar="C",
        help=f"Nozzle setpoint deg C (default {DEFAULT_NOZZLE_TEMP_C:g})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Watch layers without sending G-code",
    )
    args = parser.parse_args()

    if args.print_speed <= 0.0:
        print("--print-speed must be positive.")
        return 1

    api_key = load_octoprint_api_key()
    if not api_key and not args.dry_run:
        print("Set PRUSASAM_OCTOPRINT_API_KEY or create api_key.py for layer detection.")
        return 1

    print("\nFixed Control — hold speed/nozzle (no logging)")
    print(f"Print speed:  {args.print_speed:g} mm/s")
    print(f"Nozzle temp:  {args.nozzle_temp:g} C")
    print("Fan:          100% (locked)  |  Flow: 100%  |  Dwell: 0")

    if not args.dry_run and not test_connections():
        return 1

    octoprint_url = DEFAULT_OCTOPRINT_URL
    if not args.dry_run:
        octoprint_url = read_printer_temps()["octoprint_url"] or DEFAULT_OCTOPRINT_URL

    if not args.dry_run:
        input("  Press Enter when the print is running...")

    run_fixed_control(
        octoprint_url,
        api_key,
        print_speed_mms=float(args.print_speed),
        nozzle_temp_c=float(args.nozzle_temp),
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
