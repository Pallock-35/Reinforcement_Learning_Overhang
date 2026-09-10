from __future__ import annotations

import unittest

from print_run_naming import (
    format_angle_slug,
    normalize_print_run_key,
    parse_print_number_from_run_key,
    parse_print_run_folder_parts,
    print_run_folder_name,
    print_run_photos_dir,
)


class PrintRunNamingTests(unittest.TestCase):
    def test_print_run_folder_name_schema(self) -> None:
        self.assertEqual(
            print_run_folder_name(1, control_mode="mpc", print_angle_deg=60.0),
            "print_001_mpc_a60",
        )
        self.assertEqual(
            print_run_folder_name(2, control_mode="fixed", print_angle_deg=60.0),
            "print_002_fixed_a60",
        )
        self.assertEqual(
            print_run_folder_name(3, control_mode="mpc_temp", print_angle_deg=62.5),
            "print_003_mpc_temp_a62p5",
        )

    def test_legacy_short_folder_name(self) -> None:
        self.assertEqual(print_run_folder_name(4), "print_004")

    def test_parse_new_and_legacy_folder_names(self) -> None:
        legacy = parse_print_run_folder_parts("print_001")
        self.assertIsNotNone(legacy)
        self.assertEqual(legacy.print_number, 1)
        self.assertIsNone(legacy.control_mode)

        named = parse_print_run_folder_parts("print_007_mpc_temp_a65")
        self.assertIsNotNone(named)
        self.assertEqual(named.print_number, 7)
        self.assertEqual(named.control_mode, "mpc_temp")
        self.assertAlmostEqual(named.print_angle_deg or 0.0, 65.0)

    def test_normalize_print_run_key(self) -> None:
        self.assertEqual(normalize_print_run_key("print 2"), "print_002")
        self.assertEqual(
            normalize_print_run_key("print_003_mpc_a60"),
            "print_003_mpc_a60",
        )
        self.assertEqual(
            normalize_print_run_key("print_004_mpc_temp_a70"),
            "print_004_mpc_temp_a70",
        )

    def test_parse_print_number_from_run_key(self) -> None:
        self.assertEqual(parse_print_number_from_run_key("print_005_fixed_a55"), 5)

    def test_format_angle_slug(self) -> None:
        self.assertEqual(format_angle_slug(60.0), "a60")
        self.assertEqual(format_angle_slug(62.5), "a62p5")

    def test_print_run_photos_dir_uses_schema(self) -> None:
        path = print_run_photos_dir(
            1,
            runs_root="print_runs",
            control_mode="mpc",
            print_angle_deg=70.0,
        )
        self.assertEqual(
            str(path).replace("\\", "/"),
            "print_runs/print_001_mpc_a70/photos",
        )


if __name__ == "__main__":
    unittest.main()
