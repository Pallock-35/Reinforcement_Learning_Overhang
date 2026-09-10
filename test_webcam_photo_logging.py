from __future__ import annotations

import tempfile
import sys
import types
import unittest
from datetime import datetime
from pathlib import Path

from PIL import Image


class _FakeWidget:
    pass


tkinter_module = types.ModuleType("tkinter")
tkinter_module.Tk = _FakeWidget
tkinter_module.Misc = _FakeWidget
tkinter_module.Event = _FakeWidget
tkinter_module.Canvas = _FakeWidget
tkinter_module.Text = _FakeWidget
tkinter_module.BooleanVar = _FakeWidget
tkinter_ttk_module = types.ModuleType("tkinter.ttk")
tkinter_ttk_module.LabelFrame = _FakeWidget
image_tk_module = types.ModuleType("PIL.ImageTk")
image_tk_module.PhotoImage = _FakeWidget
api_key_module = types.ModuleType("api_key")
api_key_module.API_key = "placeholder"
sys.modules.setdefault("tkinter", tkinter_module)
sys.modules.setdefault("tkinter.ttk", tkinter_ttk_module)
sys.modules.setdefault("PIL.ImageTk", image_tk_module)
sys.modules.setdefault("api_key", api_key_module)

from prusa_dashboard import WebcamStreamer


class WebcamPhotoLoggingTests(unittest.TestCase):
    def test_logs_photo_every_configured_interval(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            streamer = WebcamStreamer(
                "http://example.invalid/webcam",
                photo_log_path=Path(temp_dir),
                photo_log_interval_seconds=15.0,
            )
            image = Image.new("RGB", (4, 4), color=(255, 0, 0))
            timestamp = datetime(2026, 6, 17, 6, 30, 0)

            first = streamer._maybe_log_photo(image, now_monotonic=100.0, timestamp=timestamp)
            skipped = streamer._maybe_log_photo(image, now_monotonic=110.0, timestamp=timestamp)
            second = streamer._maybe_log_photo(image, now_monotonic=115.0, timestamp=timestamp)

            self.assertIsNotNone(first)
            self.assertIsNone(skipped)
            self.assertIsNotNone(second)
            self.assertNotEqual(first, second)
            self.assertTrue(first and first.exists())
            self.assertTrue(second and second.exists())
            self.assertEqual(first.name, "webcam_20260617_063000.jpg")
            self.assertEqual(second.name, "webcam_20260617_063000_01.jpg")

            with Image.open(first) as saved:
                self.assertEqual(saved.size, (4, 4))
                self.assertEqual(saved.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
