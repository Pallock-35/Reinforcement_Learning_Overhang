#!/usr/bin/env python3
"""Fit thermal model from calibration print logs (outline §5)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from mpc_controller import MPCConfig, short_horizon_prediction_error
from thermal_model import (
    ThermalModelParams,
    analytical_layer_step,
    fit_thermal_model,
    model_params_from_dict,
    model_params_to_dict,
)


def load_calibration_rows(log_path: Path) -> list[dict[str, float]]:
    """Extract calibration rows from an extended thermal log.csv."""
    rows: list[dict[str, float]] = []
    with log_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            T_k = _float_or_none(row.get("T_k") or row.get("roi_mean"))
            T_next = _float_or_none(row.get("new_roi_mean") or row.get("new_T_k"))
            if T_k is None or T_next is None:
                continue
            T_nozzle = (
                _float_or_none(row.get("nozzle_setpoint_c"))
                or _float_or_none(row.get("hotend_target"))
                or 215.0
            )
            rows.append(
                {
                    "T_k": T_k,
                    "T_ir": T_k,
                    "D_k": _float_or_none(row.get("D_k")) or 0.0,
                    "v": _float_or_none(row.get("print_speed_mms")) or 100.0,
                    "T_nozzle": T_nozzle,
                    "alpha_deg": _float_or_none(row.get("print_angle_deg")) or 60.0,
                    "T_next_measured": T_next,
                }
            )
    return rows


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def evaluate_fit(
    rows: list[dict[str, float]],
    model: ThermalModelParams,
    cfg: MPCConfig,
    *,
    horizon: int = 3,
) -> dict[str, float]:
    """One-step RMSE/R² plus short-horizon e_N (outline §5.4)."""
    errors = []
    for row in rows:
        T_nozzle = float(row.get("T_nozzle", 215.0))
        T_pred, _, _ = analytical_layer_step(
            T=row["T_k"],
            u=[row["v"], T_nozzle],
            alpha_deg=row["alpha_deg"],
            model=model,
            cfg=cfg,
            n_substeps=cfg.n_substeps,
        )
        errors.append(T_pred - row["T_next_measured"])
    err = np.array(errors, dtype=float)
    ss_res = float(np.sum(err**2))
    y = np.array([row["T_next_measured"] for row in rows], dtype=float)
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) if len(y) > 1 else 0.0
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(err**2))) if len(err) else float("nan")

    # Build a sequential series for multi-step rollout when possible.
    series = []
    for row in rows:
        T_nozzle = float(row.get("T_nozzle", 215.0))
        series.append(
            {
                "T_ir": row["T_k"],
                "T_k": row["T_k"],
                "v": row["v"],
                "T_nozzle": T_nozzle,
                "alpha_deg": row["alpha_deg"],
                "T_next_measured": row["T_next_measured"],
            }
        )
        series.append(
            {
                "T_ir": row["T_next_measured"],
                "T_k": row["T_next_measured"],
                "v": row["v"],
                "T_nozzle": T_nozzle,
                "alpha_deg": row["alpha_deg"],
            }
        )
    e_n = short_horizon_prediction_error(series, model, cfg, horizon=horizon)

    return {
        "rmse": rmse,
        "r2": r2,
        "e_N": e_n,
        "horizon_N": float(horizon),
        "n_rows": float(len(rows)),
    }


def run_calibration(
    log_paths: list[Path],
    *,
    base_model: ThermalModelParams | None = None,
    output_dir: Path,
    horizon: int = 3,
) -> dict[str, Any]:
    cfg = MPCConfig()
    base = base_model or ThermalModelParams()

    all_rows: list[dict[str, float]] = []
    for path in log_paths:
        all_rows.extend(load_calibration_rows(path))

    if len(all_rows) < 5:
        raise ValueError(f"Need at least 5 calibration rows; got {len(all_rows)}")

    fitted = fit_thermal_model(all_rows, base, cfg)
    metrics = evaluate_fit(all_rows, fitted, cfg, horizon=horizon)

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "fitted_params": model_params_to_dict(fitted),
        "metrics": metrics,
        "source_logs": [str(p) for p in log_paths],
    }
    report_path = output_dir / "fit_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    params_path = output_dir / "thermal_fit.json"
    params_path.write_text(json.dumps(report["fitted_params"], indent=2), encoding="utf-8")

    print(f"Calibration rows: {len(all_rows)}")
    print(f"RMSE: {metrics['rmse']:.3f} deg C")
    print(f"R²:   {metrics['r2']:.4f}")
    print(f"e_N:  {metrics['e_N']:.3f} deg C (N={horizon})")
    print(f"Saved {report_path}")
    print(f"Saved {params_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit thermal model from calibration logs")
    parser.add_argument(
        "logs",
        nargs="+",
        type=Path,
        help="One or more thermal_layer_logger log.csv paths",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("print_runs/calibration_fit"),
        help="Directory for fit_report.json and thermal_fit.json",
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        help="Optional JSON with starting ThermalModelParams",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=3,
        help="Short-horizon prediction length N for e_N (default 3)",
    )
    args = parser.parse_args()

    base = None
    if args.base_model and args.base_model.exists():
        base = model_params_from_dict(json.loads(args.base_model.read_text(encoding="utf-8")))

    try:
        run_calibration(
            args.logs,
            base_model=base,
            output_dir=args.output_dir,
            horizon=args.horizon,
        )
    except (OSError, ValueError) as error:
        print(f"Calibration failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
