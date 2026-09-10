from __future__ import annotations

import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import dashboard_bridge


class FakeThermalFeed:
    def __init__(self, snapshot: dict[str, Any] | None) -> None:
        self._snapshot = snapshot

    def snapshot(self) -> dict[str, Any] | None:
        return self._snapshot


class DashboardBridgeApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_fetch_printer_status = dashboard_bridge.fetch_printer_status
        self._original_send_gcode_commands = dashboard_bridge.send_gcode_commands
        self._original_thermal_feed = dashboard_bridge.THERMAL_FEED
        self._original_rl_status = dashboard_bridge._RL_STATUS
        dashboard_bridge._RL_STATUS = None
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), dashboard_bridge.DashboardHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self._base_url = f"http://127.0.0.1:{self._server.server_port}"

    def tearDown(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=1)
        dashboard_bridge.fetch_printer_status = self._original_fetch_printer_status
        dashboard_bridge.send_gcode_commands = self._original_send_gcode_commands
        dashboard_bridge.THERMAL_FEED = self._original_thermal_feed
        dashboard_bridge._RL_STATUS = self._original_rl_status

    def test_temperatures_endpoint_returns_bed_temperature(self) -> None:
        dashboard_bridge.fetch_printer_status = self._fake_printer_status

        response = self._get_json("/api/temperatures")

        self.assertTrue(response["ok"])
        self.assertEqual(response["bed_temperature"], {"actual": 60.5, "target": 65.0, "offset": 0.0})
        self.assertEqual(response["temperatures"]["tool0"], {"actual": 212.25, "target": 215.0, "offset": None})

    def test_telemetry_endpoint_returns_thermal_matrix_and_bed_temperature(self) -> None:
        dashboard_bridge.fetch_printer_status = self._fake_printer_status
        dashboard_bridge.THERMAL_FEED = FakeThermalFeed(
            {
                "source": "thermal-pi",
                "connected": True,
                "updated_at": "2026-06-09 20:57:00",
                "error_text": "",
                "rows": 2,
                "cols": 3,
                "min": 24.0,
                "max": 31.5,
                "values": [[24.0, 25.5, 26.0], [29.0, 30.0, 31.5]],
            }
        )

        response = self._get_json("/api/telemetry")

        self.assertTrue(response["ok"])
        self.assertEqual(response["bed_temperature"]["actual"], 60.5)
        self.assertEqual(response["thermal_rows"], 2)
        self.assertEqual(response["thermal_cols"], 3)
        self.assertEqual(response["thermal_matrix"], [[24.0, 25.5, 26.0], [29.0, 30.0, 31.5]])

    def test_gcode_endpoint_forwards_command_to_octoprint(self) -> None:
        calls: list[tuple[str, str, list[str]]] = []

        def fake_send_gcode_commands(base_url: str, api_key: str, commands: list[str]) -> int:
            calls.append((base_url, api_key, commands))
            return 204

        dashboard_bridge.send_gcode_commands = fake_send_gcode_commands

        response = self._post_json("/api/gcode", {"command": " M105 "})

        self.assertEqual(response, {"ok": True, "octoprint_status": 204, "sent": ["M105"]})
        self.assertEqual(calls, [(dashboard_bridge.OCTOPRINT_URL, dashboard_bridge.API_key, ["M105"])])

    def test_gcode_endpoint_rejects_empty_commands(self) -> None:
        with self.assertRaises(HTTPError) as context:
            self._post_json("/api/gcode", {"commands": ["M105", ""]})

        self.assertEqual(context.exception.code, 400)
        payload = json.loads(context.exception.read().decode("utf-8"))
        self.assertFalse(payload["ok"])
        self.assertIn("must not be empty", payload["error"])

    def test_gcode_endpoint_rejects_empty_command_batch(self) -> None:
        with self.assertRaises(HTTPError) as context:
            self._post_json("/api/gcode", {"commands": []})

        self.assertEqual(context.exception.code, 400)
        payload = json.loads(context.exception.read().decode("utf-8"))
        self.assertFalse(payload["ok"])
        self.assertIn("at least one", payload["error"])

    def test_dashboard_endpoint_returns_speed_from_feedrate(self) -> None:
        dashboard_bridge.fetch_printer_status = self._fake_printer_status_with_feedrate

        response = self._get_json("/api/dashboard")

        self.assertTrue(response["connected"])
        self.assertEqual(response["speed"]["feedrate_percent"], 85.0)
        self.assertEqual(response["speed"]["speed_mms"], 144.5)
        self.assertEqual(response["speed"]["base_speed_mms"], dashboard_bridge.BASE_SPEED_MMS)
        self.assertEqual(response["speed"]["source"], "octoprint")

    def test_dashboard_prefers_rl_layer_logger_speed(self) -> None:
        dashboard_bridge.fetch_printer_status = self._fake_printer_status_with_feedrate
        self._post_json(
            "/api/rl-status",
            {
                "active": True,
                "speed_mms": 120.0,
                "base_speed_mms": 170.0,
                "print_number": 2,
                "layer": 15,
            },
        )

        response = self._get_json("/api/dashboard")

        self.assertEqual(response["speed"]["source"], "rl_layer_logger")
        self.assertEqual(response["speed"]["speed_mms"], 120.0)
        self.assertEqual(response["speed"]["print_number"], 2)
        self.assertEqual(response["speed"]["layer"], 15)
        self.assertTrue(response["speed"]["rl_active"])

    def test_rl_status_clear_marks_inactive(self) -> None:
        self._post_json("/api/rl-status", {"active": True, "speed_mms": 100.0, "print_number": 1})
        self._post_json("/api/rl-status", {"active": False})

        response = self._get_json("/api/rl-status")
        self.assertFalse(response["active"])
        self.assertIsNone(response["status"])

    def _get_json(self, path: str) -> dict[str, Any]:
        with urlopen(f"{self._base_url}{path}", timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            f"{self._base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _fake_printer_status(base_url: str, api_key: str) -> dict[str, Any]:  # noqa: ARG004
        return {
            "temperature": {
                "bed": {"actual": 60.5, "target": 65, "offset": 0},
                "tool0": {"actual": 212.25, "target": 215, "offset": None},
                "chamber": {"actual": 36, "target": None, "offset": None},
            },
            "state": {"text": "Operational", "flags": {"printing": False}},
        }

    @staticmethod
    def _fake_printer_status_with_feedrate(base_url: str, api_key: str) -> dict[str, Any]:  # noqa: ARG004
        payload = DashboardBridgeApiTests._fake_printer_status(base_url, api_key)
        payload["state"]["flags"]["feedrate"] = 85
        return payload


if __name__ == "__main__":
    unittest.main()
