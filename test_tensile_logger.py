import csv
import gzip
import json
import sqlite3
from pathlib import Path
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
from tensile_logger import GcodeIndex, allocate_run, main, thermal_fields, resolve_condition, reserve_specimen_ids, gcode_condition


class TensileTests(unittest.TestCase):
    def test_gcode_conditions_override_stale_fan_label_and_infer_other_factors(self):
        plan = json.loads(Path(__file__).with_name('tensile_experiment_plan.json').read_text())
        settings = dict(plan['baseline']['extrusion_speed_settings'])
        settings.update(min_fan_speed='10', max_fan_speed='10', temperature='260', extrusion_multiplier='1')
        self.assertEqual(gcode_condition('auto', settings, plan)[0], 'baseline')
        fan = dict(settings, min_fan_speed='30', max_fan_speed='30')
        self.assertEqual(gcode_condition('auto', fan, plan)[0], 'fan30')
        self.assertEqual(gcode_condition('fan5', fan, plan)[0], 'fan30')
        self.assertEqual(gcode_condition('baseline', fan, plan)[0], 'fan30')
        self.assertEqual(gcode_condition('auto', dict(settings, temperature='250'), plan)[0], 'nozzle250')
        self.assertEqual(gcode_condition('auto', dict(settings, extrusion_multiplier='0.95'), plan)[0], 'flow095')
        slower = dict(settings)
        for key, value in plan['baseline']['extrusion_speed_settings'].items():
            try:
                slower[key] = str(float(value) * .6)
            except ValueError:
                pass  # Relative feature speeds inherit their parent speed.
        self.assertEqual(gcode_condition('auto', slower, plan)[0], 'speed60')
        with self.assertRaises(ValueError):
            gcode_condition('auto', dict(fan, temperature='250'), plan)
        with self.assertRaises(ValueError):
            gcode_condition('nozzle250', settings, plan)
        with self.assertRaises(ValueError):
            gcode_condition('auto', {'min_fan_speed': '30'}, plan)

    def test_explicit_reuse_requires_released_failed_specimen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, _ = allocate_run(root, 'fan0', None)
            self.assertEqual(reserve_specimen_ids(root, first, 1, 'fan0', 1), ['T001'])
            second, _ = allocate_run(root, 'fan5', None)
            with self.assertRaises(ValueError):
                reserve_specimen_ids(root, second, 1, 'fan5', 1, ['T001'])
            with sqlite3.connect(root / 'specimen_registry.sqlite3') as db:
                db.execute('INSERT INTO released_specimens VALUES (?, ?, ?)', ('T001', first.name, 'failed'))
            self.assertEqual(reserve_specimen_ids(root, second, 1, 'fan5', 1, ['T001']), ['T001'])
            with sqlite3.connect(root / 'specimen_registry.sqlite3') as db:
                self.assertEqual(db.execute('SELECT condition FROM specimens WHERE specimen_id="T001"').fetchone()[0], 'fan5')
            with self.assertRaises(ValueError):
                reserve_specimen_ids(root, first, 1, 'fan0', 2, ['T001'])

    def test_short_labels_persist_across_conditions_and_registry_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, _ = allocate_run(root, 'baseline', None)
            labels = reserve_specimen_ids(root, first, 2, 'baseline', 1)
            self.assertEqual(labels, ['T001', 'T002'])
            (first / 'metadata.json').write_text(json.dumps({'specimen_ids': labels}))
            second, _ = allocate_run(root, 'fan30', None)
            self.assertEqual(reserve_specimen_ids(root, second, 1, 'fan30', 1), ['T003'])
            (second / 'metadata.json').write_text(json.dumps({'specimen_ids': ['T003']}))
            with (root / 'specimen_index.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[-1]['condition'], 'fan30')
            self.assertEqual(rows[-1]['specimen_id'], 'T003')
            (root / 'specimen_registry.sqlite3').unlink()
            third, _ = allocate_run(root, 'baseline', None)
            self.assertEqual(reserve_specimen_ids(root, third, 1, 'baseline', 2), ['T004'])

    def test_condition_labels_do_not_silently_default_to_baseline(self):
        plan = {'conditions': {'fan30': {'factor': 'fan', 'level': '30 percent'}}}
        self.assertEqual(resolve_condition('fan30', None, None, plan), ('fan', '30 percent'))
        self.assertEqual(resolve_condition('fan30', 'fan', '40 percent', plan), ('fan', '40 percent'))
        with self.assertRaises(ValueError):
            resolve_condition('unknown', None, None, plan)
        with self.assertRaises(ValueError):
            resolve_condition('fan30', 'speed', None, plan)

    def test_byte_positions_and_commands(self):
        prefix = '; µ specimen\r\nM107\r\n;LAYER_CHANGE\r\n;Z:0.2\r\n'.encode()
        fan = b'M106 S25.5\n'
        data = prefix + fan + b'M106 P1 S255\nM109 R260\nM190 S100\nG1 X1 F6000\n; temperature = 260\n'
        index = GcodeIndex(data)
        self.assertEqual(index.at(len(prefix))['gcode_fan_percent'], 0)
        self.assertEqual(index.at(len(prefix) + len(fan) - 1)['gcode_fan_percent'], 0)
        end = index.at(len(data))
        self.assertEqual(end['gcode_fan_percent'], 10)
        self.assertEqual(end['gcode_nozzle_target_c'], 260)
        self.assertEqual(end['gcode_bed_target_c'], 100)
        self.assertEqual(end['gcode_feedrate_mm_s'], 100)
        self.assertEqual(end['gcode_layer'], 1)
        self.assertEqual(index.settings['temperature'], '260')
        self.assertEqual(index.at(None), {})

    def test_units_and_objects(self):
        data = b'M486 S0\nM486 ATest specimen\nG20\nG1 F60\n'
        index = GcodeIndex(data)
        self.assertEqual(index.objects, {'0': 'Test specimen'})
        self.assertEqual(index.at(len(data))['gcode_feedrate_mm_s'], 25.4)

    def test_unique_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, _ = allocate_run(root, 'baseline', None)
            (first / 'keep').write_text('original')
            second, number = allocate_run(root, 'baseline', None)
            self.assertEqual(number, 2)
            self.assertNotEqual(first, second)
            with self.assertRaises(ValueError):
                allocate_run(root, 'baseline', 1)
            with self.assertRaises(ValueError):
                allocate_run(root, '../bad', None)
            self.assertEqual((first / 'keep').read_text(), 'original')

    def test_stale_thermal_not_used(self):
        thermal = {'connected': True, 'values': [[42]], 'roi': {'roi_mean': 42},
                   'updated_at': (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat()}
        fields = thermal_fields({'thermal': thermal}, 3)
        self.assertFalse(fields['thermal_valid'])
        self.assertNotIn('roi_mean', fields)
        thermal['updated_at'] = datetime.now(timezone.utc).isoformat()
        self.assertEqual(thermal_fields({'thermal': thermal}, 3)['roi_mean'], 42)

    def test_end_to_end_read_only_capture(self):
        gcode = b'; filament_type = ABS\n;LAYER_CHANGE\nM106 S25.5\n'
        calls = []
        class Handler(BaseHTTPRequestHandler):
            job_calls = 0
            def log_message(self, *args):
                pass
            def do_GET(self):
                calls.append(('GET', self.path))
                if self.path == '/api/job':
                    Handler.job_calls += 1
                    payload = {'state': 'Printing' if Handler.job_calls < 3 else 'Operational',
                               'job': {'file': {'origin': 'local', 'path': 'test.gcode'}},
                               'progress': {'filepos': len(gcode), 'completion': 100}}
                elif self.path == '/downloads/files/local/test.gcode':
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(gcode)
                    return
                elif self.path == '/api/printer':
                    payload = {'temperature': {'tool0': {'actual': 259, 'target': 260}}}
                else:
                    payload = {'thermal': {'connected': True, 'updated_at': datetime.now(timezone.utc).isoformat(),
                                           'values': [[42]], 'roi': {'roi_mean': 42}}}
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode())
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                base = f'http://127.0.0.1:{server.server_port}'
                args = ['tensile_logger.py', '--condition', 'baseline', '--runs-root', tmp,
                        '--interval', '0.001', '--octoprint-url', base, '--bridge-url', base]
                with patch('sys.argv', args), patch.dict('os.environ', {'PRUSASAM_OCTOPRINT_API_KEY': 'test'}):
                    self.assertEqual(main(), 0)
                run = Path(tmp) / 'baseline_r01'
                meta = json.loads((run / 'metadata.json').read_text())
                self.assertEqual(meta['end_reason'], 'print_left_active_state')
                self.assertEqual(meta['slicer_settings']['filament_type'], 'ABS')
                self.assertEqual(meta['specimen_ids'], ['T001', 'T002'])
                self.assertEqual((run / 'source.gcode').read_bytes(), gcode)
                with (run / 'stream.csv').open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[0]['hotend_actual_c'], '259')
                self.assertEqual(rows[0]['gcode_fan_percent'], '10.0')
                with gzip.open(run / 'snapshots.jsonl.gz', 'rt') as handle:
                    snapshots = [json.loads(line) for line in handle]
                self.assertEqual(snapshots[0]['thermal']['thermal']['values'], [[42]])
                self.assertTrue(all(method == 'GET' for method, _ in calls))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
