from __future__ import annotations

import csv
import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from thermal_dose import ThermalDoseTracker
from thermal_stream_sampling import (
    STREAM_INTERVAL_S,
    create_stream_log_file,
    sample_stream_tick,
    stream_until_layer,
)


class _FakeWatcher:
  def __init__(self, layers: list[int], *, min_elapsed_s: float = 0.0) -> None:
    self._layers = list(layers)
    self._min_elapsed_s = min_elapsed_s
    self._started = time.monotonic()
    self._finished = False

  def try_pop_layer(self, timeout: float = 0.0) -> int | None:
    if self._layers and time.monotonic() - self._started >= self._min_elapsed_s:
      return self._layers.pop(0)
    time.sleep(timeout)
    return None

  def is_print_finished(self) -> bool:
    return self._finished and not self._layers


class ThermalStreamSamplingTests(unittest.TestCase):
    def test_sample_stream_tick_integrates_dose(self) -> None:
        tracker = ThermalDoseTracker(T_g=60.0)

        def read_thermal() -> dict[str, float]:
            return {"roi_mean": 80.0, "roi_max": 82.0, "roi_std": 1.0, "roi_min": 78.0}

        state = sample_stream_tick(tracker, 1.0, read_thermal)
        self.assertAlmostEqual(state.T_k, 80.0)
        self.assertAlmostEqual(state.D_k, 20.0)

        state2 = sample_stream_tick(tracker, 1.0, read_thermal)
        self.assertAlmostEqual(state2.dT_dt, 0.0)
        self.assertAlmostEqual(state2.D_k, 40.0)

    def test_stream_until_layer_logs_rows(self) -> None:
        tracker = ThermalDoseTracker(T_g=60.0)
        watcher = _FakeWatcher([1], min_elapsed_s=0.05)
        rows: list[list[str]] = []
        buffer = io.StringIO()
        writer = csv.writer(buffer)

        thermal = {
            "roi_mean": 75.0,
            "roi_max": 76.0,
            "roi_std": 0.5,
            "roi_min": 74.0,
            "frame_max": 180.0,
            "frame_mean": 55.0,
        }

        with patch(
            "thermal_stream_sampling.STREAM_INTERVAL_S",
            0.02,
        ):
            layer, state = stream_until_layer(
                watcher,
                dose_tracker=tracker,
                stream_writer=writer,
                read_thermal_summary=lambda: thermal,
                read_printer_temps=lambda: {"hotend_actual": 210.0},
                print_number=1,
                print_angle_deg=60.0,
                control_mode="mpc",
                last_decision_layer=0,
                decision_monotonic=time.monotonic() - 0.1,
                print_speed_mms=170.0,
                nozzle_setpoint_c=215.0,
                interval_s=0.02,
            )

        self.assertEqual(layer, 1)
        self.assertGreater(state.samples_since_decision, 0)
        parsed = list(csv.reader(io.StringIO(buffer.getvalue())))
        self.assertGreaterEqual(len(parsed), 1)
        self.assertAlmostEqual(float(parsed[0][7]), 75.0)

    def test_create_stream_log_file_writes_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            handle, writer, path = create_stream_log_file(tmp)
            try:
                writer.writerow(["x"])
                handle.flush()
                self.assertTrue(Path(path).exists())
            finally:
                handle.close()


if __name__ == "__main__":
    unittest.main()
