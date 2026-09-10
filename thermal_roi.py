from __future__ import annotations

import os
from typing import Any

import numpy as np

THERMAL_ROWS = 62
THERMAL_COLS = 80

# Minimum rows to skip from the top of the frame (hotend band in camera view).
ROI_SKIP_TOP_MIN = int(os.environ.get("PRUSASAM_ROI_SKIP_TOP_MIN", "28"))
# Fallback start row when auto-detection cannot find a clear deposit band.
ROI_SKIP_TOP_FALLBACK = int(os.environ.get("PRUSASAM_ROI_SKIP_TOP_FALLBACK", "30"))
# Fixed deposit ROI box (stable on screen; does not shrink to warm pixels).
ROI_BOX_ROWS = int(os.environ.get("PRUSASAM_ROI_BOX_ROWS", "1"))
ROI_BOX_COLS = int(os.environ.get("PRUSASAM_ROI_BOX_COLS", "4"))
ROI_COL_CENTER_FALLBACK = int(
    os.environ.get("PRUSASAM_ROI_COL_CENTER_FALLBACK", str(THERMAL_COLS // 2))
)
# Legacy patch size kept for tests and dashboard fallbacks.
ROI_N_ROWS = int(os.environ.get("PRUSASAM_ROI_N_ROWS", str(ROI_BOX_ROWS)))
ROI_N_COLS = int(os.environ.get("PRUSASAM_ROI_N_COLS", str(ROI_BOX_COLS)))
ROI_ROW_GAP = int(os.environ.get("PRUSASAM_ROI_ROW_GAP", "0"))
# Minimum rows to leave between the nozzle tip (frame hottest row) and the ROI box top.
ROI_BELOW_HOTTEST_GAP = int(os.environ.get("PRUSASAM_ROI_BELOW_HOTTEST_GAP", "1"))
ROI_SEARCH_ROWS = int(os.environ.get("PRUSASAM_ROI_SEARCH_ROWS", str(ROI_BOX_ROWS)))
ROI_SEARCH_COLS = int(os.environ.get("PRUSASAM_ROI_SEARCH_COLS", str(ROI_BOX_COLS)))

# Pixels at or above this temperature are treated as hotend, not print surface.
HOTEND_PIXEL_MAX_C = float(os.environ.get("PRUSASAM_HOTEND_PIXEL_MAX_C", "90"))
# Warm tail used when locating the nozzle column.
HOTEND_TRIANGLE_WARM_C = float(os.environ.get("PRUSASAM_HOTEND_TRIANGLE_WARM_C", "60"))
# IR hotend triangle: wide at top (hot block), tip pointing down.
HOTEND_TRIANGLE_H = int(os.environ.get("PRUSASAM_HOTEND_TRIANGLE_H", "3"))
HOTEND_TRIANGLE_W = int(os.environ.get("PRUSASAM_HOTEND_TRIANGLE_W", "7"))
# Rows above the deposit line used to locate the nozzle column (reduces jitter).
HOTEND_COL_BAND_ROWS = int(os.environ.get("PRUSASAM_HOTEND_COL_BAND_ROWS", "4"))
# Filament must be at least this many deg C above the estimated background.
FILAMENT_ABOVE_BG_C = float(os.environ.get("PRUSASAM_FILAMENT_ABOVE_BG_C", "8"))
# Drop the hottest fraction of filament pixels before summarising temperature.
ROI_TRIM_TOP_FRAC = float(os.environ.get("PRUSASAM_ROI_TRIM_TOP_FRAC", "0.10"))
MIN_FILAMENT_PIXELS = int(os.environ.get("PRUSASAM_MIN_FILAMENT_PIXELS", "3"))
# Row stabilizer: longer dwell at one Y makes row changes harder (X is unaffected).
ROI_ROW_STABILIZE_BASE_FRAMES = int(os.environ.get("PRUSASAM_ROI_ROW_STABILIZE_BASE_FRAMES", "5"))
ROI_ROW_STABILIZE_DWELL_STEP = int(os.environ.get("PRUSASAM_ROI_ROW_STABILIZE_DWELL_STEP", "10"))
ROI_ROW_STABILIZE_MAX_FRAMES = int(os.environ.get("PRUSASAM_ROI_ROW_STABILIZE_MAX_FRAMES", "45"))


class RoiRowStabilizer:
    """Resist vertical ROI jumps; the longer a row is held, the stickier it becomes."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.stable_row: int | None = None
        self.dwell_frames = 0
        self.candidate_row: int | None = None
        self.candidate_frames = 0

    def _switch_threshold(self) -> int:
        if self.stable_row is None:
            return 1
        extra = self.dwell_frames // max(1, ROI_ROW_STABILIZE_DWELL_STEP)
        return min(ROI_ROW_STABILIZE_MAX_FRAMES, ROI_ROW_STABILIZE_BASE_FRAMES + extra)

    def stabilize(self, proposed_row: int) -> int:
        if self.stable_row is None:
            self.stable_row = proposed_row
            self.dwell_frames = 1
            self.candidate_row = None
            self.candidate_frames = 0
            return proposed_row

        if proposed_row == self.stable_row:
            self.dwell_frames += 1
            self.candidate_row = None
            self.candidate_frames = 0
            return self.stable_row

        if proposed_row == self.candidate_row:
            self.candidate_frames += 1
        else:
            self.candidate_row = proposed_row
            self.candidate_frames = 1

        if self.candidate_frames >= self._switch_threshold():
            self.stable_row = proposed_row
            self.dwell_frames = 1
            self.candidate_row = None
            self.candidate_frames = 0
        return self.stable_row

    def force(self, row: int) -> int:
        self.stable_row = row
        self.dwell_frames = 1
        self.candidate_row = None
        self.candidate_frames = 0
        return row


_ROW_STABILIZER = RoiRowStabilizer()


def reset_roi_row_stabilizer() -> None:
    """Reset temporal row smoothing (useful in tests)."""
    _ROW_STABILIZER.reset()


def detect_deposit_row_start(matrix: np.ndarray) -> int:
    """Return the first row below the top hot band (any row max >= HOTEND_PIXEL_MAX_C)."""
    rows = matrix.shape[0]
    if rows == 0:
        return 0

    row_maxes = matrix.max(axis=1)
    last_hot_row = -1
    seen_hot = False
    for row in range(rows):
        if float(row_maxes[row]) >= HOTEND_PIXEL_MAX_C:
            last_hot_row = row
            seen_hot = True
        elif seen_hot:
            return max(last_hot_row + 1, ROI_SKIP_TOP_MIN)

    if seen_hot:
        return min(max(last_hot_row + 1, ROI_SKIP_TOP_MIN), max(0, rows - ROI_BOX_ROWS))

    return min(ROI_SKIP_TOP_FALLBACK, max(0, rows - ROI_BOX_ROWS))


def detect_hotend_col_center(matrix: np.ndarray, deposit_row_start: int) -> int:
    """Return the nozzle column from the bottom of the hot band (stable frame-to-frame)."""
    cols = matrix.shape[1]
    if deposit_row_start <= 0 or cols == 0:
        return min(cols - 1, max(0, ROI_COL_CENTER_FALLBACK))

    band_top = max(0, deposit_row_start - HOTEND_COL_BAND_ROWS)
    hotend_rows = matrix[band_top:deposit_row_start, :]
    mask = hotend_rows >= HOTEND_PIXEL_MAX_C
    if not mask.any():
        warm_mask = hotend_rows >= HOTEND_TRIANGLE_WARM_C
        if warm_mask.any():
            hot_cols = np.where(warm_mask)[1]
            return int(np.median(hot_cols))
        return min(cols - 1, max(0, ROI_COL_CENTER_FALLBACK))

    hot_cols = np.where(mask)[1]
    return int(np.median(hot_cols))


def estimate_background(matrix: np.ndarray, box_row_end: int) -> float:
    """Estimate ambient/bed temperature from cold pixels below the ROI box."""
    rows = matrix.shape[0]
    tail = matrix[box_row_end:, :]
    if tail.size >= 20:
        cold = tail.ravel()
    else:
        cold = matrix[matrix < HOTEND_PIXEL_MAX_C].ravel()
    if cold.size == 0:
        return 30.0
    return float(np.percentile(cold, 20))


def hottest_row_bottom(matrix: np.ndarray) -> int:
    """Return the lowest row that contains the frame maximum temperature."""
    if matrix.size == 0:
        return 0
    frame_max = float(np.max(matrix))
    hot_rows, _ = np.where(matrix >= frame_max)
    if hot_rows.size:
        return int(hot_rows.max())
    return int(np.argmax(matrix.max(axis=1)))


def box_overlaps_hottest(matrix: np.ndarray, row_start: int, row_end: int) -> bool:
    """True when the proposed ROI rows touch the frame's hottest pixel(s)."""
    if matrix.size == 0 or row_start >= row_end:
        return False
    frame_max = float(np.max(matrix))
    hot_rows, _ = np.where(matrix >= frame_max)
    if hot_rows.size == 0:
        return False
    return bool(np.any((hot_rows >= row_start) & (hot_rows < row_end)))


def min_row_below_nozzle(matrix: np.ndarray) -> int:
    """First allowed ROI start row strictly below the nozzle tip (frame hottest)."""
    rows = matrix.shape[0]
    if rows == 0:
        return 0
    return min(hottest_row_bottom(matrix) + 1 + ROI_BELOW_HOTTEST_GAP, max(0, rows - 1))


def box_is_below_nozzle(matrix: np.ndarray, row_start: int) -> bool:
    """True when the ROI top is at or below the nozzle-floor row."""
    return row_start >= min_row_below_nozzle(matrix)


def _clamp_box_below_hottest(matrix: np.ndarray, row_start: int, row_end: int) -> tuple[int, int]:
    """Ensure the box starts strictly below the nozzle tip (frame hottest row)."""
    rows = matrix.shape[0]
    floor = min_row_below_nozzle(matrix)
    if row_start >= floor:
        return row_start, row_end
    row_start = floor
    row_end = min(row_start + ROI_BOX_ROWS, rows)
    return row_start, row_end


def deposit_box_bounds(matrix: np.ndarray) -> tuple[int, int, int, int, int]:
    """Return a fixed ROI box hugging the nozzle and covering its sides."""
    rows = matrix.shape[0]
    hot_band_end = detect_deposit_row_start(matrix)
    proposed_start = min(hot_band_end + ROI_ROW_GAP, max(0, rows - 1))
    proposed_end = min(proposed_start + ROI_BOX_ROWS, rows)
    proposed_start, proposed_end = _clamp_box_below_hottest(matrix, proposed_start, proposed_end)

    row_start = _ROW_STABILIZER.stabilize(proposed_start)
    row_end = min(row_start + ROI_BOX_ROWS, rows)
    if not box_is_below_nozzle(matrix, row_start) or box_overlaps_hottest(matrix, row_start, row_end):
        forced_start, _ = _clamp_box_below_hottest(matrix, row_start, row_end)
        row_start = _ROW_STABILIZER.force(forced_start)
        row_end = min(row_start + ROI_BOX_ROWS, rows)

    hot_col = detect_hotend_col_center(matrix, row_start)
    cols = matrix.shape[1]
    half = ROI_BOX_COLS // 2
    col_start = max(0, hot_col - half)
    col_end = min(cols, col_start + ROI_BOX_COLS)
    col_start = max(0, col_end - ROI_BOX_COLS)
    return row_start, row_end, col_start, col_end, hot_col


def _hotend_triangle_mask(
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    hot_col: int,
) -> np.ndarray:
    """Mark pixels inside the nozzle triangle at the top of the ROI box."""
    height = row_end - row_start
    width = col_end - col_start
    mask = np.zeros((height, width), dtype=bool)
    if height == 0 or width == 0:
        return mask

    tri_height = min(HOTEND_TRIANGLE_H, height)
    tri_base_width = HOTEND_TRIANGLE_W
    base_half = tri_base_width // 2
    for tri_row in range(tri_height):
        if tri_height <= 1:
            half_w = 0
        else:
            half_w = int(round(base_half * (tri_height - 1 - tri_row) / (tri_height - 1)))
        for col in range(hot_col - half_w, hot_col + half_w + 1):
            if col < col_start or col >= col_end:
                continue
            mask[tri_row, col - col_start] = True
    return mask


def _filament_stats_mask(
    matrix: np.ndarray,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    hot_col: int,
) -> np.ndarray:
    """Pixels in the fixed box that count toward filament temperature."""
    roi = matrix[row_start:row_end, col_start:col_end]
    if roi.size == 0:
        return np.zeros((0, 0), dtype=bool)

    background = estimate_background(matrix, row_end)
    min_temp = background + FILAMENT_ABOVE_BG_C
    triangle = _hotend_triangle_mask(row_start, row_end, col_start, col_end, hot_col)
    hotend = roi >= HOTEND_PIXEL_MAX_C
    cold = roi < min_temp
    stats = ~(triangle | hotend | cold)

    if int(stats.sum()) < MIN_FILAMENT_PIXELS:
        stats = ~(triangle | hotend)
    if int(stats.sum()) < MIN_FILAMENT_PIXELS:
        stats = ~triangle
    return stats


def _trim_hot_pixels(pixels: np.ndarray) -> np.ndarray:
    """Drop the hottest fraction of filament pixels before summarising."""
    if pixels.size <= 1 or ROI_TRIM_TOP_FRAC <= 0:
        return pixels
    cutoff = float(np.percentile(pixels, (1.0 - ROI_TRIM_TOP_FRAC) * 100.0))
    trimmed = pixels[pixels <= cutoff]
    return trimmed if trimmed.size else pixels


def deposit_row_bounds(matrix: np.ndarray) -> tuple[int, int]:
    row_start, row_end, _, _, _ = deposit_box_bounds(matrix)
    return row_start, row_end


def deposit_col_bounds(matrix: np.ndarray, row_start: int) -> tuple[int, int]:
    _, _, col_start, col_end, _ = deposit_box_bounds(matrix)
    return col_start, col_end


def deposit_bounds(matrix: np.ndarray) -> tuple[int, int, int, int]:
    row_start, row_end, col_start, col_end, _ = deposit_box_bounds(matrix)
    return row_start, row_end, col_start, col_end


def deposit_pixels(
    matrix: np.ndarray,
) -> tuple[np.ndarray, int, int, int, int]:
    """Return filament stats pixels and the fixed ROI box bounds."""
    row_start, row_end, col_start, col_end, hot_col = deposit_box_bounds(matrix)
    if row_start >= row_end or col_start >= col_end:
        return np.array([], dtype=np.float32), row_start, row_end, col_start, col_end

    mask = _filament_stats_mask(matrix, row_start, row_end, col_start, col_end, hot_col)
    roi = matrix[row_start:row_end, col_start:col_end]
    pixels = roi[mask].astype(np.float32, copy=False)
    return pixels, row_start, row_end, col_start, col_end


def summarize_deposit_roi(matrix: np.ndarray) -> dict[str, Any]:
    """Summarise extruded-filament temperature in a fixed box around the nozzle."""
    row_start, row_end, col_start, col_end, hot_col = deposit_box_bounds(matrix)
    pixels, box_row_start, box_row_end, box_col_start, box_col_end = deposit_pixels(matrix)
    if pixels.size == 0:
        return {
            "roi_max": float("nan"),
            "roi_mean": float("nan"),
            "roi_std": float("nan"),
            "roi_min": float("nan"),
            "roi_median": float("nan"),
            "deposit_temp_c": float("nan"),
            "frame_max": float(np.max(matrix)) if matrix.size else float("nan"),
            "frame_mean": float(np.mean(matrix)) if matrix.size else float("nan"),
            "roi_row_start": box_row_start,
            "roi_row_end": box_row_end,
            "roi_col_start": box_col_start,
            "roi_col_end": box_col_end,
            "roi_skip_top": box_row_start,
            "roi_n_rows": max(0, box_row_end - box_row_start),
            "roi_n_cols": max(0, box_col_end - box_col_start),
            "filament_pixel_count": 0,
            "background_temp_c": float("nan"),
            "hotend_col": hot_col,
            "hotend_triangle_h": HOTEND_TRIANGLE_H,
            "hotend_triangle_w": HOTEND_TRIANGLE_W,
            "roi_trim_top_frac": ROI_TRIM_TOP_FRAC,
            "hotend_pixel_max_c": HOTEND_PIXEL_MAX_C,
        }

    background = estimate_background(matrix, row_end)
    stats_pixels = _trim_hot_pixels(pixels)
    deposit_core = float(np.percentile(stats_pixels, 90)) if stats_pixels.size else float("nan")
    return {
        "roi_max": float(np.max(stats_pixels)),
        "roi_mean": float(np.mean(stats_pixels)),
        "roi_std": float(np.std(stats_pixels)),
        "roi_min": float(np.min(stats_pixels)),
        "roi_median": float(np.median(stats_pixels)),
        "deposit_temp_c": deposit_core,
        "frame_max": float(np.max(matrix)),
        "frame_mean": float(np.mean(matrix)),
        "roi_row_start": box_row_start,
        "roi_row_end": box_row_end,
        "roi_col_start": box_col_start,
        "roi_col_end": box_col_end,
        "roi_skip_top": box_row_start,
        "roi_n_rows": box_row_end - box_row_start,
        "roi_n_cols": box_col_end - box_col_start,
        "filament_pixel_count": int(pixels.size),
        "background_temp_c": background,
        "hotend_col": hot_col,
        "hotend_triangle_h": HOTEND_TRIANGLE_H,
        "hotend_triangle_w": HOTEND_TRIANGLE_W,
        "roi_trim_top_frac": ROI_TRIM_TOP_FRAC,
        "hotend_pixel_max_c": HOTEND_PIXEL_MAX_C,
    }


def attach_roi_to_thermal_snapshot(snapshot: dict[str, Any] | None) -> dict[str, Any] | None:
    """Add deposit ROI stats to a thermal feed snapshot dict."""
    if snapshot is None:
        return None
    values = snapshot.get("values")
    if not values:
        return snapshot

    matrix = np.array(values, dtype=np.float32)
    enriched = dict(snapshot)
    enriched["roi"] = summarize_deposit_roi(matrix)
    return enriched

# Back-compat alias used in older call sites during refactors.
deposit_search_bounds = deposit_box_bounds
