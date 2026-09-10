from __future__ import annotations

import unittest

from analyze_thermal_runs import compute_run_metrics
from thermal_targets import (
    DELTA_D,
    dose_target,
    dose_window,
    temperature_bond_limit,
    temperature_sag_limit,
    temperature_target,
)


class ThermalTargetsTests(unittest.TestCase):
    def test_dose_window_uses_absolute_delta(self) -> None:
        d_star = dose_target(55.0)
        low, high = dose_window(55.0)
        self.assertAlmostEqual(d_star - low, DELTA_D, places=6)
        self.assertAlmostEqual(high - d_star, DELTA_D, places=6)

    def test_targets_decrease_with_overhang_angle(self) -> None:
        self.assertGreater(temperature_target(45.0), temperature_target(70.0))
        self.assertGreater(dose_target(45.0), dose_target(70.0))
        self.assertGreater(temperature_bond_limit(45.0), temperature_bond_limit(70.0))
        self.assertGreater(temperature_sag_limit(45.0), temperature_sag_limit(70.0))

    def test_sag_limit_above_bond_limit(self) -> None:
        self.assertGreater(temperature_sag_limit(60.0), temperature_bond_limit(60.0))


class AnalyzeMetricsTests(unittest.TestCase):
    def test_tracking_errors_are_mae_not_mse(self) -> None:
        rows = [
            {
                "print_angle_deg": "45",
                "T_k": "147",
                "T_star": "145",
                "D_k": "732",
                "D_star": "730",
                "D_min": "550",
                "D_max": "910",
                "print_speed_mms": "35",
                "fan_percent": "40",
                "dwell_s": "1",
                "layer_dt_s": "10",
            },
            {
                "print_angle_deg": "45",
                "T_k": "143",
                "T_star": "145",
                "D_k": "728",
                "D_star": "730",
                "D_min": "550",
                "D_max": "910",
                "print_speed_mms": "40",
                "fan_percent": "50",
                "dwell_s": "0",
                "layer_dt_s": "12",
            },
        ]
        metrics = compute_run_metrics(rows)
        self.assertAlmostEqual(metrics["E_T"], 2.0, places=6)
        self.assertAlmostEqual(metrics["E_D"], 2.0, places=6)
        self.assertEqual(metrics["V_D"], 0.0)
        self.assertAlmostEqual(metrics["t_print"], 22.0, places=6)


if __name__ == "__main__":
    unittest.main()
