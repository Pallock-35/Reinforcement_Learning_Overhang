from __future__ import annotations

import unittest

from printer_control import BASE_SPEED_MMS, build_control_gcode


class PrinterControlTests(unittest.TestCase):
    def test_build_control_gcode_speed_and_nozzle(self) -> None:
        commands = build_control_gcode(34.0, 215.0)
        self.assertIn("M220 S20", commands)
        self.assertIn("M221 S100", commands)
        self.assertIn("M106 S255", commands)
        self.assertIn("M104 S215", commands)
        self.assertFalse(any(cmd.startswith("G4") for cmd in commands))

    def test_build_control_gcode_fan_always_full(self) -> None:
        commands = build_control_gcode(BASE_SPEED_MMS, 200.0)
        self.assertIn("M106 S255", commands)
        self.assertNotIn("M107", commands)
        self.assertIn("M104 S200", commands)

    def test_speed_percent_clamped(self) -> None:
        commands = build_control_gcode(10.0, 210.0)
        self.assertIn("M220 S10", commands)

    def test_nozzle_temp_clamped(self) -> None:
        low = build_control_gcode(BASE_SPEED_MMS, 150.0)
        high = build_control_gcode(BASE_SPEED_MMS, 260.0)
        self.assertIn("M104 S190", low)
        self.assertIn("M104 S230", high)


if __name__ == "__main__":
    unittest.main()
