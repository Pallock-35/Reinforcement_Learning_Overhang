from __future__ import annotations

import unittest

from thermal_dose import ThermalDoseTracker


class ThermalDoseTests(unittest.TestCase):
    def test_dose_accumulation_identity(self) -> None:
        tracker = ThermalDoseTracker(T_g=60.0)
        D1, dT1 = tracker.update(80.0, 10.0)
        self.assertEqual(dT1, 0.0)
        self.assertAlmostEqual(D1, 10.0 * (80.0 - 60.0))

        D2, dT2 = tracker.update(70.0, 5.0)
        self.assertAlmostEqual(dT2, (70.0 - 80.0) / 5.0)
        self.assertAlmostEqual(D2, D1 + 5.0 * (70.0 - 60.0))

    def test_sub_threshold_temperature_adds_no_dose(self) -> None:
        tracker = ThermalDoseTracker(T_g=60.0)
        D, _ = tracker.update(50.0, 12.0)
        self.assertEqual(D, 0.0)

    def test_reset_clears_state(self) -> None:
        tracker = ThermalDoseTracker(T_g=60.0)
        tracker.update(90.0, 5.0)
        tracker.reset()
        self.assertEqual(tracker.D, 0.0)
        self.assertIsNone(tracker.T_prev)


if __name__ == "__main__":
    unittest.main()
