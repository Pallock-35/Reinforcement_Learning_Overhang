from __future__ import annotations

import csv
import io
import os
import queue
import time
from collections import deque
from datetime import datetime, timedelta
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk
from typing import Any
from urllib.request import urlopen, Request

from PIL import Image
from PIL import ImageTk
import queue as _queue


WEBCAM_PHOTO_LOG_PATH = Path(__file__).resolve().parent / "webcam_photo_log"
WEBCAM_PHOTO_LOG_INTERVAL_SECONDS = 15.0


class WebcamStreamer:
    """Simple MJPEG streamer that keeps the latest JPEG bytes for the UI.

    The streamer reads the MJPEG multipart stream and stores the most recent
    JPEG frame as raw bytes. The UI thread converts bytes -> ImageTk.PhotoImage.
    """

    def __init__(
        self,
        url: str,
        target_size: tuple[int, int] = (640, 360),
        photo_log_path: Path | str = WEBCAM_PHOTO_LOG_PATH,
        photo_log_interval_seconds: float = WEBCAM_PHOTO_LOG_INTERVAL_SECONDS,
    ) -> None:
        self.url = url
        self._lock = threading.Lock()
        # single-slot queue holding the most recent PIL.Image
        self._queue: _queue.Queue[Image.Image] = _queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._target_size = target_size
        self._photo_log_path = Path(photo_log_path)
        self._photo_log_interval_seconds = photo_log_interval_seconds
        self._last_photo_log_at: float | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def get_latest_image(self) -> Image.Image | None:
        try:
            return self._queue.get_nowait()
        except _queue.Empty:
            return None

    def _set_last_image(self, img: Image.Image) -> None:
        try:
            self._queue.put_nowait(img)
        except _queue.Full:
            try:
                _ = self._queue.get_nowait()
            except _queue.Empty:
                pass
            try:
                self._queue.put_nowait(img)
            except _queue.Full:
                # give up if we race
                pass

    def _maybe_log_photo(
        self,
        img: Image.Image,
        now_monotonic: float | None = None,
        timestamp: datetime | None = None,
    ) -> Path | None:
        if self._photo_log_interval_seconds <= 0:
            return None

        now = time.monotonic() if now_monotonic is None else now_monotonic
        if self._last_photo_log_at is not None and now - self._last_photo_log_at < self._photo_log_interval_seconds:
            return None

        self._last_photo_log_at = now
        captured_at = timestamp or datetime.now()
        filename = f"webcam_{captured_at.strftime('%Y%m%d_%H%M%S')}.jpg"
        path = self._photo_log_path / filename
        suffix = 1
        while path.exists():
            path = self._photo_log_path / f"webcam_{captured_at.strftime('%Y%m%d_%H%M%S')}_{suffix:02d}.jpg"
            suffix += 1

        try:
            self._photo_log_path.mkdir(parents=True, exist_ok=True)
            img.save(path, format="JPEG", quality=90)
        except Exception:
            return None
        return path

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                req = Request(self.url, headers={"User-Agent": "PrusaSAM/1.0"})
                with urlopen(req, timeout=10) as resp:
                    buf = b""
                    while not self._stop.is_set():
                        chunk = resp.read(1024)
                        if not chunk:
                            break
                        buf += chunk
                        start = buf.find(b"\xff\xd8")
                        end = buf.find(b"\xff\xd9")
                        if start != -1 and end != -1 and end > start:
                            jpg = buf[start : end + 2]
                            buf = buf[end + 2 :]
                            try:
                                img = Image.open(io.BytesIO(jpg)).convert("RGB")
                                self._maybe_log_photo(img)
                                display_img = img.resize(self._target_size, Image.Resampling.LANCZOS)
                                self._set_last_image(display_img)
                            except Exception:
                                # ignore individual frame errors
                                pass
            except Exception:
                # transient error; wait then reconnect
                if self._stop.wait(0.5):
                    break
                continue

from octoprint_sensor_check import (
    OCTOPRINT_URL,
    API_key,
    fetch_printer_status,
)


REFRESH_INTERVAL_MS = 3000
REQUEST_POLL_MS = 200
TEMPERATURE_HISTORY_MINUTES = 30

# Keep a small, curated set of temperature sensor keys that are meaningful
# for display. This workspace only cares about the bed and primary toolhead.
USEFUL_TEMPERATURE_KEYS = {"bed", "tool0"}


@dataclass(frozen=True)
class TemperatureReading:
    actual: float | None
    target: float | None
    offset: float | None


class RollingTemperatureGraph(ttk.LabelFrame):
    def __init__(self, master: tk.Misc, title: str) -> None:
        super().__init__(master, text=title, padding=10)
        self._window = timedelta(minutes=TEMPERATURE_HISTORY_MINUTES)
        self._samples: deque[tuple[datetime, float | None, float | None]] = deque()
        self._title = title

        header = ttk.Frame(self)
        header.pack(fill="x", pady=(0, 6))
        self._summary_text = tk.StringVar(value="Waiting for temperature data...")
        ttk.Label(header, textvariable=self._summary_text, style="Subtle.TLabel").pack(anchor="w")

        self._canvas = tk.Canvas(
            self,
            height=170,
            background="#101418",
            highlightthickness=1,
            highlightbackground="#1f2933",
        )
        self._canvas.pack(fill="both", expand=True)
        self._canvas.bind("<Configure>", self._redraw)

    def record(self, when: datetime, actual: float | None, target: float | None) -> None:
        self._samples.append((when, actual, target))
        self._prune(when)
        self._update_summary(actual, target)
        self._redraw()

    def _update_summary(self, actual: float | None, target: float | None) -> None:
        if actual is None and target is None:
            self._summary_text.set("No sample yet")
            return
        if target is None:
            self._summary_text.set(f"Current: {self._format_temp(actual)}")
            return
        self._summary_text.set(f"Current: {self._format_temp(actual)}   Target: {self._format_temp(target)}")

    @staticmethod
    def _format_temp(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{value:.1f} °C"

    def _prune(self, when: datetime) -> None:
        cutoff = when - self._window
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def _visible_samples(self) -> list[tuple[datetime, float | None, float | None]]:
        return list(self._samples)

    def _redraw(self, event: tk.Event | None = None) -> None:  # noqa: ARG002 - Tk callback signature
        canvas = self._canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 2)
        height = max(canvas.winfo_height(), 2)

        self._draw_axes(canvas, width, height)

        samples = self._visible_samples()
        if not samples:
            canvas.create_text(
                width / 2,
                height / 2,
                text="Waiting for temperature data...",
                fill="#94a3b8",
                font=("Segoe UI", 10),
            )
            return

        values = [value for _, actual, target in samples for value in (actual, target) if value is not None]
        if not values:
            return

        y_min = min(values)
        y_max = max(values)
        if y_min == y_max:
            y_min -= 5.0
            y_max += 5.0
        padding = max((y_max - y_min) * 0.12, 3.0)
        y_min -= padding
        y_max += padding
        y_min = 0.0

        left_pad = 50
        right_pad = 16
        top_pad = 14
        bottom_pad = 30
        plot_left = left_pad
        plot_right = width - right_pad
        plot_top = top_pad
        plot_bottom = height - bottom_pad
        plot_width = max(plot_right - plot_left, 1)
        plot_height = max(plot_bottom - plot_top, 1)
        now = samples[-1][0]
        window_seconds = self._window.total_seconds()
        start = now - self._window

        self._draw_y_labels(canvas, plot_left, plot_top, plot_bottom, y_min, y_max)
        self._draw_x_labels(canvas, plot_left, plot_top, plot_bottom)

        def project(when: datetime, value: float) -> tuple[float, float]:
            x_ratio = max(0.0, min((when - start).total_seconds() / window_seconds, 1.0))
            y_ratio = (value - y_min) / (y_max - y_min)
            x = plot_left + x_ratio * plot_width
            y = plot_bottom - y_ratio * plot_height
            return x, y

        self._draw_series(canvas, samples, project, "actual", "#f59e0b")
        self._draw_series(canvas, samples, project, "target", "#22c55e", dashed=True)

    def _draw_axes(self, canvas: tk.Canvas, width: int, height: int) -> None:
        plot_left = 50
        plot_right = width - 16
        plot_top = 14
        plot_bottom = height - 30
        canvas.create_rectangle(plot_left, plot_top, plot_right, plot_bottom, outline="#334155")
        canvas.create_text(
            plot_right,
            plot_bottom + 14,
            text="0 min",
            fill="#94a3b8",
            anchor="e",
            font=("Segoe UI", 8),
        )
        canvas.create_text(
            plot_left,
            plot_bottom + 14,
            text=f"{TEMPERATURE_HISTORY_MINUTES} min",
            fill="#94a3b8",
            anchor="w",
            font=("Segoe UI", 8),
        )
        canvas.create_text(
            plot_left,
            plot_top - 2,
            text=self._title,
            fill="#cbd5e1",
            anchor="sw",
            font=("Segoe UI", 9, "bold"),
        )

    def _draw_y_labels(
        self,
        canvas: tk.Canvas,
        plot_left: int,
        plot_top: int,
        plot_bottom: int,
        y_min: float,
        y_max: float,
    ) -> None:
        ticks = 4
        for index in range(ticks + 1):
            ratio = index / ticks
            y = plot_bottom - ratio * (plot_bottom - plot_top)
            value = y_min + ratio * (y_max - y_min)
            canvas.create_line(plot_left, y, canvas.winfo_width() - 16, y, fill="#1f2937")
            canvas.create_text(
                plot_left - 6,
                y,
                text=f"{value:.0f}",
                fill="#94a3b8",
                anchor="e",
                font=("Segoe UI", 8),
            )

    def _draw_x_labels(self, canvas: tk.Canvas, plot_left: int, plot_top: int, plot_bottom: int) -> None:
        width = canvas.winfo_width()
        labels = [(0, "0"), (10, "10"), (20, "20"), (30, "30")]
        for minutes, label in labels:
            ratio = minutes / TEMPERATURE_HISTORY_MINUTES
            x = plot_left + (width - plot_left - 16) * (1.0 - ratio)
            canvas.create_line(x, plot_top, x, plot_bottom, fill="#1f2937")
            canvas.create_text(
                x,
                plot_bottom + 14,
                text=f"{label} min",
                fill="#94a3b8",
                anchor="center",
                font=("Segoe UI", 8),
            )

    def _draw_series(
        self,
        canvas: tk.Canvas,
        samples: list[tuple[datetime, float | None, float | None]],
        project: Any,
        field: str,
        color: str,
        dashed: bool = False,
    ) -> None:
        points: list[tuple[float, float]] = []
        for when, actual, target in samples:
            value = actual if field == "actual" else target
            if value is None:
                continue
            x, y = project(when, value)
            points.append((x, y))

        if len(points) == 1:
            x, y = points[0]
            canvas.create_oval(x - 1, y - 1, x + 1, y + 1, fill=color, outline=color)
            return

        if len(points) > 1:
            flattened: list[float] = []
            for x, y in points:
                flattened.extend((x, y))
            line_kwargs: dict[str, Any] = {
                "fill": color,
                "width": 1,
                "smooth": True,
                "splinesteps": 16,
            }
            if dashed:
                line_kwargs["dash"] = (4, 3)
            canvas.create_line(*flattened, **line_kwargs)
            for x, y in points:
                canvas.create_oval(x - 1, y - 1, x + 1, y + 1, fill=color, outline=color)


def _parse_temperature(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


@dataclass(frozen=True)
class DashboardSnapshot:
    state_text: str
    temperature_lines: list[str]
    temperatures: dict[str, TemperatureReading]
    sample_time: datetime
    updated_at: str
    is_printing: bool = False
    error_text: str = ""


class PrinterDashboard(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("PrusaSAM Dashboard")
        self.geometry("1120x900")
        self.minsize(980, 760)

        self._result_queue: queue.Queue[DashboardSnapshot] = queue.Queue()
        self._worker_lock = threading.Lock()
        self._worker_running = False
        self._auto_refresh = tk.BooleanVar(value=True)
        self._status_text = tk.StringVar(value="Ready")
        self._updated_text = tk.StringVar(value="Never")
        self._state_text = tk.StringVar(value="Waiting for data...")
        # Webcam streamer
        self._webcam_url = f"{OCTOPRINT_URL.rstrip('/')}/webcam/?action=stream"
        self._webcam = WebcamStreamer(self._webcam_url, target_size=(640, 360))
        self._webcam.start()

        self._log_fh: tuple[Any, Any] | None = None
        self._logging = False
        self._csv_path = ""

        self._build_ui()
        self._start_logging()
        self.after(100, self._poll_results)
        self.after(250, self.refresh)
        # Periodic webcam update: poll the latest frame queue frequently
        self.after(33, self._update_webcam)
        

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Subtle.TLabel", foreground="#666666")
        style.configure("Status.TLabel", font=("Segoe UI", 10, "bold"))

        root = ttk.Frame(self, padding=16)
        root.pack(fill="both", expand=True)

        top = ttk.Frame(root)
        top.pack(fill="x")

        ttk.Label(top, text="PrusaSAM Dashboard", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            top,
            text=f"Live OctoPrint data from {OCTOPRINT_URL}",
            style="Subtle.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        controls = ttk.Frame(root)
        controls.pack(fill="x", pady=(14, 8))

        ttk.Checkbutton(
            controls,
            text="Auto refresh",
            variable=self._auto_refresh,
            command=self._schedule_refresh,
        ).pack(side="left")
        ttk.Button(controls, text="Refresh now", command=self.refresh).pack(side="left", padx=(10, 0))
        ttk.Label(controls, textvariable=self._status_text, style="Status.TLabel").pack(side="right")

        info = ttk.LabelFrame(root, text="Info", padding=12)
        info.pack(fill="x")

        ttk.Label(info, textvariable=self._state_text, wraplength=900, justify="left").pack(anchor="w")
        ttk.Label(info, text="Last update:", style="Status.TLabel").pack(anchor="w", pady=(8, 2))
        ttk.Label(info, textvariable=self._updated_text).pack(anchor="w")

        self._error_text = tk.StringVar(value="")
        ttk.Label(info, textvariable=self._error_text, foreground="#9a2f2f", wraplength=900).pack(anchor="w", pady=(8, 0))

        ttk.Label(info, text="Temperature sensors:", style="Status.TLabel").pack(anchor="w", pady=(10, 4))
        self._sensor_box = tk.Text(info, height=4, wrap="none", borderwidth=0, highlightthickness=0)
        self._sensor_box.pack(fill="x")
        self._sensor_box.configure(state="disabled")

        body = ttk.Frame(root)
        body.pack(fill="both", expand=True, pady=(8, 0))

        graphs = ttk.LabelFrame(body, text="Temperature History", padding=12)
        graphs.pack(side="left", fill="both", expand=True, padx=(0, 8))

        self._bed_graph = RollingTemperatureGraph(graphs, "Bed")
        self._bed_graph.pack(fill="both", expand=True)
        self._hotend_graph = RollingTemperatureGraph(graphs, "Hotend")
        self._hotend_graph.pack(fill="both", expand=True, pady=(8, 0))

        webcam_frame = ttk.LabelFrame(body, text="Webcam", padding=8)
        webcam_frame.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self._webcam_label = ttk.Label(webcam_frame)
        self._webcam_label.pack(fill="both", expand=True)

        status = ttk.Frame(root)
        status.pack(fill="x", pady=(10, 0))
        ttk.Separator(status).pack(fill="x", pady=(0, 8))
        ttk.Label(status, textvariable=self._status_text).pack(anchor="w")

    def _schedule_refresh(self) -> None:
        if self._auto_refresh.get():
            self.after(REFRESH_INTERVAL_MS, self.refresh)

    def refresh(self) -> None:
        with self._worker_lock:
            if self._worker_running:
                return
            self._worker_running = True

        self._status_text.set("Refreshing...")
        self._error_text.set("")
        thread = threading.Thread(target=self._load_snapshot, daemon=True)
        thread.start()

    def _load_snapshot(self) -> None:
        try:
            payload = fetch_printer_status(OCTOPRINT_URL, API_key)
            state = payload.get("state")
            is_printing = False
            if isinstance(state, dict):
                flags = state.get("flags") or {}
                is_printing = bool(flags.get("printing", False)) or ("print" in state.get("text", "").lower())
            sample_time = datetime.now()
            snapshot = DashboardSnapshot(
                state_text=self._extract_state(payload),
                temperature_lines=self._extract_temperatures(payload),
                temperatures=self._extract_temperature_readings(payload),
                sample_time=sample_time,
                updated_at=sample_time.strftime("%Y-%m-%d %H:%M:%S"),
                is_printing=is_printing,
            )
        except Exception as error:  # noqa: BLE001 - show errors in the dashboard
            sample_time = datetime.now()
            snapshot = DashboardSnapshot(
                state_text="No data",
                temperature_lines=[],
                temperatures={},
                sample_time=sample_time,
                updated_at=sample_time.strftime("%Y-%m-%d %H:%M:%S"),
                error_text=str(error),
            )
        self._result_queue.put(snapshot)

    def _poll_results(self) -> None:
        try:
            snapshot = self._result_queue.get_nowait()
        except queue.Empty:
            self.after(REQUEST_POLL_MS, self._poll_results)
            return

        self._apply_snapshot(snapshot)
        with self._worker_lock:
            self._worker_running = False

        if self._auto_refresh.get():
            self.after(REFRESH_INTERVAL_MS, self.refresh)
        self.after(REQUEST_POLL_MS, self._poll_results)

    def _apply_snapshot(self, snapshot: DashboardSnapshot) -> None:
        self._state_text.set(snapshot.state_text)
        self._updated_text.set(snapshot.updated_at)
        self._status_text.set("Updated")
        self._error_text.set(snapshot.error_text)
        self._set_sensor_lines(snapshot.temperature_lines)
        self._record_temperature_sample(snapshot)

    def _set_sensor_lines(self, lines: list[str]) -> None:
        self._sensor_box.configure(state="normal")
        self._sensor_box.delete("1.0", "end")
        if lines:
            self._sensor_box.insert("1.0", "\n".join(lines))
        else:
            self._sensor_box.insert("1.0", "No temperature sensor data returned.")
        self._sensor_box.configure(state="disabled")

    def _fast_position_poll(self) -> None:
        # Position polling removed; keep this hook no-op in case of scheduling.
        return

    def _on_close(self) -> None:
        try:
            self._webcam.stop()
        finally:
            self.destroy()

    def _update_webcam(self) -> None:
        try:
            img = self._webcam.get_latest_image()
            if img is not None:
                photo = ImageTk.PhotoImage(img)
                # keep a reference to prevent GC
                setattr(self._webcam_label, "image", photo)
                self._webcam_label.configure(image=photo)
        except Exception:
            pass
        self.after(33, self._update_webcam)

    def _start_logging(self) -> None:
        if self._logging:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"prusa_temps_{ts}.csv"
        try:
            fh = open(fname, "w", newline="", encoding="utf-8")
            writer = csv.writer(fh)
            writer.writerow(["timestamp", "bed_actual", "bed_target", "hotend_actual", "hotend_target"])
            fh.flush()
        except Exception as error:
            self._log_fh = None
            self._logging = False
            self._csv_path = ""
            self._status_text.set(f"CSV logging unavailable: {error}")
            return

        self._log_fh = (fh, writer)
        self._logging = True
        self._csv_path = os.path.abspath(fname)
        self._status_text.set(f"Logging temps to {self._csv_path}")

    def _stop_logging(self) -> None:
        if not self._logging:
            return
        try:
            log_fh = self._log_fh
            if log_fh is None:
                return
            fh, _ = log_fh
            fh.flush()
            fh.close()
        finally:
            self._log_fh = None
            self._logging = False
            self._status_text.set("Logging stopped")

    def _append_log(self, snapshot: DashboardSnapshot) -> None:
        if not self._logging or not self._log_fh:
            return
        fh, writer = self._log_fh
        bed = snapshot.temperatures.get("bed")
        hotend = snapshot.temperatures.get("tool0")
        writer.writerow([
            snapshot.sample_time.strftime("%Y-%m-%d %H:%M:%S"),
            "" if bed is None or bed.actual is None else f"{bed.actual:.2f}",
            "" if bed is None or bed.target is None else f"{bed.target:.2f}",
            "" if hotend is None or hotend.actual is None else f"{hotend.actual:.2f}",
            "" if hotend is None or hotend.target is None else f"{hotend.target:.2f}",
        ])
        fh.flush()

    def _record_temperature_sample(self, snapshot: DashboardSnapshot) -> None:
        if self._logging:
            try:
                self._append_log(snapshot)
            except Exception:
                pass

        bed = snapshot.temperatures.get("bed")
        hotend = snapshot.temperatures.get("tool0")
        self._bed_graph.record(snapshot.sample_time, bed.actual if bed else None, bed.target if bed else None)
        self._hotend_graph.record(
            snapshot.sample_time,
            hotend.actual if hotend else None,
            hotend.target if hotend else None,
        )

    @staticmethod
    def _extract_state(payload: dict[str, Any]) -> str:
        state = payload.get("state")
        if isinstance(state, dict):
            return f"State: {state.get('text', 'unknown')}"
        return "State: unknown"

    @staticmethod
    def _extract_temperatures(payload: dict[str, Any]) -> list[str]:
        return [
            f"{name}: actual={reading.actual}, target={reading.target}, offset={reading.offset}"
            for name, reading in PrinterDashboard._extract_temperature_readings(payload).items()
        ]

    @staticmethod
    def _extract_temperature_readings(payload: dict[str, Any]) -> dict[str, TemperatureReading]:
        temperature = payload.get("temperature", {})
        readings: dict[str, TemperatureReading] = {}
        if not isinstance(temperature, dict):
            return readings
        for sensor_name, sensor_data in temperature.items():
            # Skip sensors not in the curated useful set. If the set is empty,
            # show all sensors (backwards-compatible behavior).
            if USEFUL_TEMPERATURE_KEYS and sensor_name not in USEFUL_TEMPERATURE_KEYS:
                continue
            if not isinstance(sensor_data, dict):
                continue
            readings[sensor_name] = TemperatureReading(
                actual=_parse_temperature(sensor_data.get("actual")),
                target=_parse_temperature(sensor_data.get("target")),
                offset=_parse_temperature(sensor_data.get("offset")),
            )
        return readings


def main() -> int:
    app = PrinterDashboard()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
