#!/usr/bin/env python3
"""Analyze thermal control runs: E_T, E_D, V_D, TV_u, t_print (outline §9)."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from thermal_targets import dose_target, dose_window, temperature_target


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def load_run_rows(log_path: Path) -> list[dict[str, Any]]:
    with log_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compute_run_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Outline §9 metrics: MAE tracking errors, window violation rate, TV_u, t_print."""
    if not rows:
        return {}

    T_vals = []
    T_star_vals = []
    D_vals = []
    D_star_vals = []
    D_low_vals = []
    D_high_vals = []
    speeds = []
    nozzles = []
    dt_vals = []

    for row in rows:
        alpha = _float(row.get("print_angle_deg"))
        T_k = _float(row.get("T_k") or row.get("new_roi_mean") or row.get("roi_mean"))
        D_k = _float(row.get("D_k"))
        T_star = _float(row.get("T_star")) or temperature_target(alpha)
        D_star = _float(row.get("D_star") or row.get("D_target")) or dose_target(alpha)
        D_min = _float(row.get("D_min") or row.get("D_low"))
        D_max = _float(row.get("D_max") or row.get("D_high"))
        if D_min == 0.0 and D_max == 0.0:
            D_min, D_max = dose_window(alpha)

        T_vals.append(T_k)
        T_star_vals.append(T_star)
        D_vals.append(D_k)
        D_star_vals.append(D_star)
        D_low_vals.append(D_min)
        D_high_vals.append(D_max)
        speeds.append(_float(row.get("print_speed_mms")))
        nozzles.append(
            _float(row.get("nozzle_setpoint_c"))
            or _float(row.get("hotend_target"), 215.0)
        )
        dt_vals.append(_float(row.get("layer_dt_s"), 30.0))

    T = np.array(T_vals, dtype=float)
    T_star = np.array(T_star_vals, dtype=float)
    D = np.array(D_vals, dtype=float)
    D_star = np.array(D_star_vals, dtype=float)
    D_low = np.array(D_low_vals, dtype=float)
    D_high = np.array(D_high_vals, dtype=float)

    violation = np.logical_or(D < D_low, D > D_high)

    # Outline §9.1 / §9.2: mean absolute tracking errors.
    E_T = float(np.mean(np.abs(T - T_star)))
    E_D = float(np.mean(np.abs(D - D_star)))
    V_D = float(np.mean(violation))
    t_print = float(np.sum(dt_vals))

    u = np.column_stack([speeds, nozzles])
    TV_u = (
        float(np.sum(np.linalg.norm(np.diff(u, axis=0), axis=1))) if len(u) > 1 else 0.0
    )

    return {
        "E_T": E_T,
        "E_D": E_D,
        "V_D": V_D,
        "TV_u": TV_u,
        "t_print": t_print,
        "n_layers": float(len(rows)),
        "mean_T": float(np.mean(T)),
        "mean_D": float(np.mean(D)),
    }


def summarize_runs(run_dirs: list[Path]) -> list[dict[str, Any]]:
    summaries = []
    for run_dir in run_dirs:
        log_path = run_dir / "log.csv"
        if not log_path.exists():
            continue
        rows = load_run_rows(log_path)
        metrics = compute_run_metrics(rows)
        if not metrics:
            continue
        control_mode = rows[0].get("control_mode", "unknown")
        angle = _float(rows[0].get("print_angle_deg"))
        summaries.append(
            {
                "run": run_dir.name,
                "control_mode": control_mode,
                "print_angle_deg": angle,
                **metrics,
            }
        )
    return summaries


def aggregate_by_mode_angle(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean metrics across replicates for each (mode, angle) cell."""
    buckets: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for item in summaries:
        key = (str(item["control_mode"]), round(float(item["print_angle_deg"]), 1))
        buckets[key].append(item)

    aggregates = []
    for (mode, angle), items in sorted(buckets.items()):
        aggregates.append(
            {
                "control_mode": mode,
                "print_angle_deg": angle,
                "n_replicates": len(items),
                "E_T": float(np.mean([x["E_T"] for x in items])),
                "E_D": float(np.mean([x["E_D"] for x in items])),
                "V_D": float(np.mean([x["V_D"] for x in items])),
                "TV_u": float(np.mean([x["TV_u"] for x in items])),
                "t_print": float(np.mean([x["t_print"] for x in items])),
            }
        )
    return aggregates


def print_summary_table(summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        print("No runs found.")
        return

    header = (
        f"{'Run':<12} {'Mode':<12} {'Angle':>6} "
        f"{'E_T':>10} {'E_D':>10} {'V_D':>8} {'TV_u':>10} {'t_print':>10}"
    )
    print(header)
    print("-" * len(header))
    for item in summaries:
        print(
            f"{item['run']:<12} {item['control_mode']:<12} {item['print_angle_deg']:>6.1f} "
            f"{item['E_T']:>10.2f} {item['E_D']:>10.2f} {item['V_D']:>8.3f} "
            f"{item['TV_u']:>10.2f} {item['t_print']:>10.1f}"
        )


def print_aggregate_table(aggregates: list[dict[str, Any]]) -> None:
    if not aggregates:
        return
    print("\nAggregated by mode × angle (outline §10 matrix):")
    header = (
        f"{'Mode':<12} {'Angle':>6} {'N':>3} "
        f"{'E_T':>10} {'E_D':>10} {'V_D':>8} {'TV_u':>10} {'t_print':>10}"
    )
    print(header)
    print("-" * len(header))
    for item in aggregates:
        print(
            f"{item['control_mode']:<12} {item['print_angle_deg']:>6.1f} {item['n_replicates']:>3} "
            f"{item['E_T']:>10.2f} {item['E_D']:>10.2f} {item['V_D']:>8.3f} "
            f"{item['TV_u']:>10.2f} {item['t_print']:>10.1f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze thermal control experiment logs")
    parser.add_argument(
        "runs_root",
        nargs="?",
        type=Path,
        default=Path("print_runs"),
        help="print_runs directory or specific run folder",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path to write metrics JSON",
    )
    args = parser.parse_args()

    root = args.runs_root
    if (root / "log.csv").exists():
        run_dirs = [root]
    else:
        run_dirs = sorted(p for p in root.glob("print_*") if p.is_dir())

    summaries = summarize_runs(run_dirs)
    print_summary_table(summaries)
    aggregates = aggregate_by_mode_angle(summaries)
    print_aggregate_table(aggregates)

    if args.json_out:
        payload = {"runs": summaries, "aggregates": aggregates}
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
