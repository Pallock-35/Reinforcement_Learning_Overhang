from __future__ import annotations

import unittest

import numpy as np

from thermal_roi import (
    FILAMENT_ABOVE_BG_C,
    HOTEND_PIXEL_MAX_C,
    HOTEND_TRIANGLE_H,
    HOTEND_TRIANGLE_W,
    ROI_BELOW_HOTTEST_GAP,
    ROI_BOX_COLS,
    ROI_BOX_ROWS,
    ROI_N_COLS,
    ROI_ROW_GAP,
    ROI_ROW_STABILIZE_BASE_FRAMES,
    ROI_SKIP_TOP_MIN,
    ROI_TRIM_TOP_FRAC,
    RoiRowStabilizer,
    _hotend_triangle_mask,
    box_is_below_nozzle,
    box_overlaps_hottest,
    deposit_box_bounds,
    deposit_pixels,
    detect_deposit_row_start,
    detect_hotend_col_center,
    hottest_row_bottom,
    min_row_below_nozzle,
    reset_roi_row_stabilizer,
    summarize_deposit_roi,
)


class ThermalRoiTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_roi_row_stabilizer()

    def _synthetic_frame(self) -> np.ndarray:
        rows, cols = 62, 80
        matrix = np.full((rows, cols), 30.0, dtype=np.float32)
        matrix[:ROI_SKIP_TOP_MIN, :] = 180.0
        matrix[:ROI_SKIP_TOP_MIN, 36:44] = 200.0
        row_start = max(ROI_SKIP_TOP_MIN + ROI_ROW_GAP, min_row_below_nozzle(matrix))
        col_start = 40 - ROI_N_COLS // 2
        col_end = col_start + ROI_N_COLS
        matrix[row_start : row_start + ROI_BOX_ROWS, col_start:col_end] = 75.0
        return matrix

    def _paint_triangle(
        self,
        matrix: np.ndarray,
        row_start: int,
        hot_col: int,
        tri_h: int,
        tri_w: int,
        temp: float,
    ) -> None:
        base_half = tri_w // 2
        for tri_row in range(tri_h):
            if tri_h <= 1:
                half_w = 0
            else:
                half_w = int(round(base_half * (tri_h - 1 - tri_row) / (tri_h - 1)))
            matrix[row_start + tri_row, hot_col - half_w : hot_col + half_w + 1] = temp

    def test_detect_deposit_row_start_skips_hotend_band(self) -> None:
        matrix = self._synthetic_frame()
        self.assertEqual(detect_deposit_row_start(matrix), ROI_SKIP_TOP_MIN)

    def test_detect_deposit_row_start_follows_90c_hot_band(self) -> None:
        matrix = np.full((62, 80), 30.0, dtype=np.float32)
        matrix[:24, :] = 120.0
        matrix[24:28, 36:44] = 98.0
        row_start = detect_deposit_row_start(matrix)
        self.assertEqual(row_start, 28)

    def test_detect_deposit_row_start_detects_90c_nozzle_tail(self) -> None:
        matrix = np.full((62, 80), 30.0, dtype=np.float32)
        matrix[26:30, 36:44] = 92.0
        row_start = detect_deposit_row_start(matrix)
        self.assertEqual(row_start, 30)

    def test_detect_hotend_col_center_tracks_hot_blob(self) -> None:
        matrix = self._synthetic_frame()
        row_start = detect_deposit_row_start(matrix)
        center = detect_hotend_col_center(matrix, row_start)
        self.assertGreaterEqual(center, 36)
        self.assertLessEqual(center, 44)

    def test_hotend_triangle_mask_wide_at_top_narrow_at_bottom(self) -> None:
        mask = _hotend_triangle_mask(28, 33, 34, 46, 40)
        self.assertEqual(int(mask[0].sum()), HOTEND_TRIANGLE_W)
        self.assertEqual(int(mask[HOTEND_TRIANGLE_H - 1].sum()), 1)
        self.assertGreater(int(mask[0].sum()), int(mask[HOTEND_TRIANGLE_H - 1].sum()))

    def test_deposit_box_follows_hot_band_when_clear_of_hottest(self) -> None:
        matrix = self._synthetic_frame()
        row_start, row_end, col_start, col_end, hot_col = deposit_box_bounds(matrix)
        expected = max(ROI_SKIP_TOP_MIN + ROI_ROW_GAP, min_row_below_nozzle(matrix))
        self.assertEqual(row_start, expected)
        self.assertTrue(box_is_below_nozzle(matrix, row_start))
        self.assertFalse(box_overlaps_hottest(matrix, row_start, row_end))
        self.assertEqual(row_end - row_start, ROI_BOX_ROWS)
        self.assertEqual(col_end - col_start, ROI_BOX_COLS)
        self.assertGreaterEqual(hot_col, 36)
        self.assertLessEqual(hot_col, 44)

    def test_deposit_pixels_use_fixed_box(self) -> None:
        matrix = self._synthetic_frame()
        pixels, row_start, row_end, col_start, col_end = deposit_pixels(matrix)
        self.assertEqual(row_end - row_start, ROI_BOX_ROWS)
        self.assertEqual(col_end - col_start, ROI_BOX_COLS)
        self.assertLess(float(np.max(pixels)), HOTEND_PIXEL_MAX_C)
        self.assertAlmostEqual(float(np.max(pixels)), 75.0, places=1)

    def test_summarize_deposit_roi_ignores_frame_hotend(self) -> None:
        matrix = self._synthetic_frame()
        summary = summarize_deposit_roi(matrix)
        self.assertGreater(summary["frame_max"], summary["roi_max"])
        self.assertAlmostEqual(summary["roi_max"], 75.0, places=1)
        self.assertEqual(summary["roi_n_rows"], ROI_BOX_ROWS)
        self.assertEqual(summary["roi_n_cols"], ROI_BOX_COLS)

    def test_hotend_triangle_excluded_but_sides_kept(self) -> None:
        matrix = np.full((62, 80), 30.0, dtype=np.float32)
        matrix[:ROI_SKIP_TOP_MIN, 38:42] = 180.0
        row_start, _, _, _, _ = deposit_box_bounds(matrix)
        self._paint_triangle(matrix, row_start, 40, HOTEND_TRIANGLE_H, HOTEND_TRIANGLE_W, 85.0)
        matrix[row_start : row_start + ROI_BOX_ROWS, 34:46] = 72.0
        summary = summarize_deposit_roi(matrix)
        self.assertLess(summary["roi_mean"], 80.0)
        self.assertAlmostEqual(summary["roi_mean"], 72.0, places=0)
        self.assertEqual(summary["roi_row_start"], row_start)
        self.assertEqual(summary["roi_n_rows"], ROI_BOX_ROWS)

    def test_cold_background_pixels_excluded(self) -> None:
        matrix = np.full((62, 80), 30.0, dtype=np.float32)
        matrix[:ROI_SKIP_TOP_MIN, 38:42] = 180.0
        row_start, _, col_start, col_end, _ = deposit_box_bounds(matrix)
        matrix[row_start, col_start:col_end] = 30.0 + FILAMENT_ABOVE_BG_C + 20.0
        matrix[row_start, max(0, col_start - 4) : col_start] = 30.0
        matrix[row_start, col_end : min(80, col_end + 4)] = 30.0
        summary = summarize_deposit_roi(matrix)
        self.assertAlmostEqual(summary["roi_mean"], 30.0 + FILAMENT_ABOVE_BG_C + 20.0, places=0)

    def test_box_always_clamped_below_nozzle_tip(self) -> None:
        matrix = np.full((62, 80), 30.0, dtype=np.float32)
        matrix[:ROI_SKIP_TOP_MIN, 38:42] = 180.0
        matrix[28, 40] = 200.0
        row_start, row_end, _, _, _ = deposit_box_bounds(matrix)
        self.assertEqual(row_start, min_row_below_nozzle(matrix))
        self.assertGreaterEqual(
            row_start,
            hottest_row_bottom(matrix) + 1 + ROI_BELOW_HOTTEST_GAP,
        )
        self.assertTrue(box_is_below_nozzle(matrix, row_start))
        self.assertFalse(box_overlaps_hottest(matrix, row_start, row_end))

    def test_box_clamped_below_nozzle_even_without_overlap(self) -> None:
        """ROI above the tip (no overlap) must still be pushed below the nozzle."""
        matrix = np.full((62, 80), 30.0, dtype=np.float32)
        matrix[20:26, 38:42] = 150.0
        matrix[32, 40] = 210.0
        row_start, row_end, _, _, _ = deposit_box_bounds(matrix)
        self.assertEqual(row_start, min_row_below_nozzle(matrix))
        self.assertEqual(row_start, 32 + 1 + ROI_BELOW_HOTTEST_GAP)
        self.assertTrue(box_is_below_nozzle(matrix, row_start))
        self.assertFalse(box_overlaps_hottest(matrix, row_start, row_end))

    def test_stabilizer_cannot_keep_roi_above_nozzle(self) -> None:
        matrix = np.full((62, 80), 30.0, dtype=np.float32)
        matrix[:28, 38:42] = 180.0
        deposit_box_bounds(matrix)
        matrix[34, 40] = 220.0
        row_start, row_end, _, _, _ = deposit_box_bounds(matrix)
        self.assertEqual(row_start, min_row_below_nozzle(matrix))
        self.assertTrue(box_is_below_nozzle(matrix, row_start))
        self.assertFalse(box_overlaps_hottest(matrix, row_start, row_end))

    def test_box_pushed_down_when_hot_tail_extends_below_band(self) -> None:
        matrix = np.full((62, 80), 30.0, dtype=np.float32)
        matrix[:28, 38:42] = 180.0
        matrix[28:31, 39:41] = 120.0
        row_start, row_end, _, _, _ = deposit_box_bounds(matrix)
        self.assertGreaterEqual(row_start, min_row_below_nozzle(matrix))
        self.assertTrue(box_is_below_nozzle(matrix, row_start))
        self.assertFalse(box_overlaps_hottest(matrix, row_start, row_end))

    def test_top_temperature_trim_ignores_outliers(self) -> None:
        matrix = np.full((62, 80), 30.0, dtype=np.float32)
        matrix[:ROI_SKIP_TOP_MIN, 38:42] = 180.0
        row_start, _, _, _, _ = deposit_box_bounds(matrix)
        matrix[row_start : row_start + ROI_BOX_ROWS, 34:46] = 70.0
        matrix[row_start + 2, 40] = 95.0
        summary = summarize_deposit_roi(matrix)
        self.assertLess(summary["roi_mean"], 75.0)
        self.assertAlmostEqual(summary["roi_mean"], 70.0, places=0)
        self.assertAlmostEqual(summary["roi_trim_top_frac"], ROI_TRIM_TOP_FRAC, places=3)

    def test_row_stabilizer_resists_brief_vertical_jumps(self) -> None:
        stabilizer = RoiRowStabilizer()
        self.assertEqual(stabilizer.stabilize(28), 28)
        for _ in range(3):
            self.assertEqual(stabilizer.stabilize(31), 28)
        for _ in range(ROI_ROW_STABILIZE_BASE_FRAMES):
            stabilizer.stabilize(31)
        self.assertEqual(stabilizer.stable_row, 31)

    def test_row_stabilizer_gets_stickier_with_dwell(self) -> None:
        sticky = RoiRowStabilizer()
        sticky.stabilize(28)
        for _ in range(30):
            sticky.stabilize(28)
        threshold = sticky._switch_threshold()
        self.assertGreater(threshold, ROI_ROW_STABILIZE_BASE_FRAMES)
        for _ in range(threshold - 1):
            sticky.stabilize(31)
        self.assertEqual(sticky.stable_row, 28)
        sticky.stabilize(31)
        self.assertEqual(sticky.stable_row, 31)


if __name__ == "__main__":
    unittest.main()
