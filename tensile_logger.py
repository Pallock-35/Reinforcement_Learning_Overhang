"""Read-only tensile print acquisition. Never sends printer commands."""
from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time
from datetime import datetime, timezone
from urllib.parse import quote
from urllib.request import Request, urlopen


def now():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def get(url, key=''):
    request = Request(url, headers={'X-Api-Key': key} if key else {})
    with urlopen(request, timeout=5) as response:
        return response.read()


def get_json(url, key=''):
    return json.loads(get(url, key))


def load_key():
    key = os.environ.get('PRUSASAM_OCTOPRINT_API_KEY', '').strip()
    if key:
        return key
    try:
        from api_key import API_key
        return API_key
    except ImportError:
        raise RuntimeError('Set PRUSASAM_OCTOPRINT_API_KEY or create api_key.py')


class GcodeIndex:
    """States indexed by completed line byte offsets, not execution timestamps."""
    def __init__(self, data):
        if b'\x00' in data or data.startswith(b'GCDE'):
            raise ValueError('Export plain .gcode rather than binary G-code for this collector')
        self.settings = {}
        self.objects = {}
        self.offsets = [0]
        self.states = [{}]
        state = {}
        offset = 0
        units = 1.0
        object_id = None
        for raw in data.splitlines(keepends=True):
            offset += len(raw)
            line = raw.decode('utf-8', errors='replace').strip()
            setting = re.match(r'^;\s*([A-Za-z_][\w]*)\s*=\s*(.*)$', line)
            if setting:
                self.settings[setting[1]] = setting[2]
            previous = dict(state)
            if line == ';LAYER_CHANGE':
                state['gcode_layer'] = state.get('gcode_layer', 0) + 1
            elif line.startswith(';Z:'):
                try:
                    state['gcode_layer_z_mm'] = float(line[3:])
                except ValueError:
                    pass
            elif line.startswith(';TYPE:'):
                state['gcode_feature'] = line[6:]
            command = re.sub(r'\([^)]*\)', '', line.split(';', 1)[0]).strip().upper()
            match = re.match(r'^(?:N\d+\s*)?([GMT]\d+)(?=\s|[A-Z]|$)', command)
            if match:
                code = match[1]
                params = {k: float(v) for k, v in re.findall(r'([A-Z])\s*(-?\d+(?:\.\d*)?)', command[match.end():])}
                if code == 'G20':
                    units = 25.4
                elif code == 'G21':
                    units = 1.0
                elif code in ('G0', 'G1', 'G2', 'G3') and 'F' in params:
                    state['gcode_feedrate_mm_s'] = params['F'] * units / 60
                elif code in ('M106', 'M107') and params.get('P', 0) == 0:
                    pwm = 0 if code == 'M107' else params.get('S', 255)
                    state['gcode_fan_percent'] = pwm / 255 * 100
                elif code in ('M104', 'M109') and params.get('T', 0) == 0:
                    value = params.get('S', params.get('R'))
                    if value is not None:
                        state['gcode_nozzle_target_c'] = value
                elif code in ('M140', 'M190'):
                    value = params.get('S', params.get('R'))
                    if value is not None:
                        state['gcode_bed_target_c'] = value
                elif code in ('M220', 'M221') and 'S' in params:
                    state['gcode_speed_factor_percent' if code == 'M220' else 'gcode_flow_factor_percent'] = params['S']
                elif code == 'M486':
                    if 'S' in params:
                        object_id = int(params['S'])
                        state['gcode_object_id'] = object_id
                    name = re.search(r'\bA(.*)', line.split(';', 1)[0])
                    if name and object_id is not None and object_id >= 0:
                        self.objects[str(object_id)] = name[1].strip()
            if state != previous:
                self.offsets.append(offset)
                self.states.append(dict(state))

    def at(self, filepos):
        if filepos is None:
            return {}
        return self.states[max(0, bisect.bisect_right(self.offsets, filepos) - 1)]


def allocate_run(root, condition, replicate):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', condition):
        raise ValueError('Condition must use letters, numbers, underscores or hyphens')
    root.mkdir(parents=True, exist_ok=True)
    number = replicate or 1
    while True:
        run = root / f'{condition}_r{number:02d}'
        try:
            run.mkdir()
            return run, number
        except FileExistsError:
            if replicate is not None:
                raise ValueError(f'{run} already exists; choose another replicate')
            number += 1


def resolve_condition(condition, factor, level, plan):
    entry = plan.get('conditions', {}).get(condition, {})
    factor = factor or entry.get('factor')
    if factor is None:
        if condition == 'baseline':
            factor = 'baseline'
        else:
            raise ValueError('Unknown condition: specify --factor and --level, or add it to the experiment plan')
    if level is None and factor == entry.get('factor'):
        level = entry.get('level')
    if factor != 'baseline' and not level:
        raise ValueError('Provide --level for the varied factor, including its units')
    return factor, level


def reserve_specimen_ids(root, run, count, condition, replicate, requested_ids=None):
    """Reserve short labels across conditions/jobs, including interrupted acquisitions."""
    root = Path(root)
    with sqlite3.connect(root / 'specimen_registry.sqlite3', timeout=30) as db:
        db.execute('BEGIN IMMEDIATE')
        db.execute('CREATE TABLE IF NOT EXISTS specimens (number INTEGER PRIMARY KEY, specimen_id TEXT UNIQUE, condition TEXT, replicate INTEGER, print_order INTEGER, run_folder TEXT)')
        db.execute('CREATE TABLE IF NOT EXISTS released_specimens (specimen_id TEXT PRIMARY KEY, previous_run TEXT, reason TEXT)')
        highest = db.execute('SELECT COALESCE(MAX(number), 0) FROM specimens').fetchone()[0]
        # Also protect labels in copied runs if the registry has been lost or replaced.
        active_labels = set()
        for meta_path in root.glob('*/metadata.json'):
            existing = json.loads(meta_path.read_text(encoding='utf-8'))
            for label in existing.get('specimen_ids', []):
                active_labels.add(label)
                match = re.fullmatch(r'T(\d+)', label)
                if match:
                    highest = max(highest, int(match[1]))
        labels = requested_ids or [f'T{highest + order:03d}' for order in range(1, count + 1)]
        if len(labels) != count or len(set(labels)) != count:
            raise ValueError('Provide one distinct specimen ID per specimen')
        for order, label in enumerate(labels, 1):
            if not re.fullmatch(r'T[0-9]{3,}', label) or int(label[1:]) < 1 or label != f'T{int(label[1:]):03d}':
                raise ValueError('Specimen IDs must have the form T001, T002, etc.')
            allocated = db.execute('SELECT 1 FROM specimens WHERE specimen_id = ?', (label,)).fetchone()
            released = db.execute('SELECT 1 FROM released_specimens WHERE specimen_id = ?', (label,)).fetchone()
            if label in active_labels or (allocated and not released):
                raise ValueError(f'{label} is already assigned; explicitly exclude its failed attempt before reuse')
            db.execute('INSERT OR REPLACE INTO specimens VALUES (?, ?, ?, ?, ?, ?)',
                       (int(label[1:]), label, condition, replicate, order, run.name))
            db.execute('DELETE FROM released_specimens WHERE specimen_id = ?', (label,))
        # Human-readable lookup: write atomically while the allocation lock is held.
        lookup = root / 'specimen_index.csv'
        temporary = root / '.specimen_index.csv.tmp'
        with temporary.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['specimen_id', 'condition', 'replicate', 'print_order', 'run_folder'])
            writer.writerows(db.execute('SELECT specimen_id, condition, replicate, print_order, run_folder FROM specimens WHERE specimen_id NOT IN (SELECT specimen_id FROM released_specimens) ORDER BY number'))
        temporary.replace(lookup)
    return labels


COLUMNS = [
    'timestamp_utc', 'elapsed_s', 'state', 'completion_percent', 'filepos',
    'print_time_s', 'print_time_left_s', 'hotend_actual_c', 'hotend_target_c',
    'bed_actual_c', 'bed_target_c', 'thermal_updated_at', 'thermal_age_s',
    'thermal_valid', 'roi_mean', 'roi_max', 'roi_min', 'roi_std', 'frame_mean', 'frame_max',
    'gcode_layer', 'gcode_layer_z_mm', 'gcode_object_id', 'gcode_feature',
    'gcode_feedrate_mm_s', 'gcode_fan_percent', 'gcode_nozzle_target_c',
    'gcode_bed_target_c', 'gcode_speed_factor_percent', 'gcode_flow_factor_percent', 'errors',
]


def thermal_fields(payload, stale_after):
    thermal = payload.get('thermal') or {}
    stamp = thermal.get('updated_at')
    age = None
    if stamp:
        try:
            # Bridge timestamps currently use the local Mac timezone.
            dt = datetime.fromisoformat(stamp)
            age = (datetime.now().astimezone() - dt.astimezone()).total_seconds()
        except (ValueError, TypeError):
            pass
    valid = bool(thermal.get('connected') and thermal.get('values') and age is not None and 0 <= age <= stale_after)
    result = {'thermal_updated_at': stamp, 'thermal_age_s': age, 'thermal_valid': valid}
    if valid:
        roi = thermal.get('roi') or {}
        for key in ('roi_mean', 'roi_max', 'roi_min', 'roi_std', 'frame_mean', 'frame_max'):
            result[key] = roi.get(key)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', required=True, help='e.g. baseline or fan30')
    parser.add_argument('--replicate', type=int, help='Automatic next unused number if omitted')
    parser.add_argument('--specimens-per-print', type=int, default=2)
    parser.add_argument('--specimen-ids', nargs='+', help='Explicit short labels; only unused or explicitly released IDs are accepted')
    parser.add_argument('--factor', choices=('baseline', 'fan', 'speed', 'nozzle', 'flow'))
    parser.add_argument('--plan', type=Path, default=Path(__file__).with_name('tensile_experiment_plan.json'))
    parser.add_argument('--level', help='Declared experimental level, e.g. 30 percent or 250 C; does not change settings')
    parser.add_argument('--material-batch', default='')
    parser.add_argument('--orientation', default='', help='Build/loading orientation, recorded as entered')
    parser.add_argument('--check-only', action='store_true', help='Inspect selected G-code and thermal connection without creating a run')
    parser.add_argument('--notes', default='')
    parser.add_argument('--octoprint-url', default=os.environ.get('PRUSASAM_OCTOPRINT_URL', 'http://cimpi-prusamk4s.local'))
    parser.add_argument('--bridge-url', default='http://127.0.0.1:8765')
    parser.add_argument('--interval', type=float, default=1.0)
    parser.add_argument('--runs-root', type=Path, default=Path('print_runs/tensile'))
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0 or args.specimens_per_print < 1 or (args.replicate is not None and args.replicate < 1):
        parser.error('Interval, replicate and specimen count must be positive')
    plan = json.loads(args.plan.read_text(encoding='utf-8')) if args.plan.exists() else {}
    try:
        args.factor, args.level = resolve_condition(args.condition, args.factor, args.level, plan)
    except ValueError as error:
        parser.error(str(error))
    key = load_key()
    base = args.octoprint_url.rstrip('/')
    job = get_json(base + '/api/job', key)
    selected = job.get('job', {}).get('file', {})
    if not selected.get('path') or selected.get('origin') != 'local':
        raise RuntimeError('Select a plain G-code file stored in OctoPrint local storage first')
    path = quote(selected['path'], safe='/')
    data = get(base + '/downloads/files/local/' + path, key)
    index = GcodeIndex(data)
    # Check thermal availability before assigning an experimental run.
    thermal = get_json(args.bridge_url.rstrip('/') + '/api/thermal')
    if not thermal_fields(thermal, max(3, args.interval * 3))['thermal_valid']:
        raise RuntimeError('Thermal feed is missing or stale; check the dashboard bridge and Pi camera server')
    print(f'Selected file: {selected.get("name", selected["path"])}', flush=True)
    print(f'Declared condition: {args.condition} | factor={args.factor} | level={args.level or "baseline"}', flush=True)
    print(f'G-code object labels: {len(index.objects)} | declared physical specimens: {args.specimens_per_print}', flush=True)
    if index.objects and len(index.objects) != args.specimens_per_print:
        print('CHECK BUILD PLATE: object labels differ from specimen count. A combined mesh can contain several specimens; verify the layout in the slicer.', flush=True)
    if args.check_only:
        fields = ('filament_type', 'layer_height', 'fill_density', 'perimeters', 'temperature',
                  'first_layer_temperature', 'bed_temperature', 'first_layer_bed_temperature',
                  'perimeter_speed', 'infill_speed', 'extrusion_multiplier', 'min_fan_speed',
                  'max_fan_speed', 'bridge_fan_speed', 'disable_fan_first_layers')
        print(json.dumps({k: index.settings.get(k) for k in fields}, indent=2))
        print('Thermal feed is fresh. No run created and no printer commands sent.')
        return 0
    run, replicate = allocate_run(args.runs_root, args.condition, args.replicate)
    try:
        specimen_ids = reserve_specimen_ids(args.runs_root, run, args.specimens_per_print, args.condition, replicate, args.specimen_ids)
    except Exception:
        run.rmdir()
        raise
    (run / 'source.gcode').write_bytes(data)
    metadata = {
        'schema_version': 1, 'experiment_type': 'tensile', 'control_mode': 'observe_only',
        'condition': args.condition, 'replicate': replicate,
        'varied_factor': args.factor, 'declared_level': args.level,
        'experiment_plan': plan,
        'material_batch': args.material_batch, 'orientation': args.orientation,
        'specimens_per_print': args.specimens_per_print,
        'declared_print_sequence': plan.get('print_sequence', 'unknown'),
        'slicer_complete_objects': index.settings.get('complete_objects'),
        'specimen_ids': specimen_ids,
        'notes': args.notes, 'created_at_utc': now(), 'octoprint_file': selected,
        'gcode_sha256': hashlib.sha256(data).hexdigest(), 'slicer_settings': index.settings,
        'gcode_objects': index.objects, 'sampling_interval_s': args.interval,
        'roi_environment': {k: v for k, v in os.environ.items() if k.startswith('PRUSASAM_ROI_')},
        'manual_context': {'ambient_temperature_c': None, 'enclosure_temperature_c': None,
                           'relative_humidity_percent': None, 'filament_drying_and_storage': '',
                           'camera_emissivity_and_calibration': '', 'conditioning_before_test': ''},
        'initial_job': job, 'joined_mid_print': job.get('state') not in ('Operational', 'Ready'),
        'limitations': [
            'G-code states follow file byte progress, which can lead buffered physical execution.',
            'G-code fan/feedrate values are commanded, not measured; manual overrides are not inferred.',
            'Thermal frames and the automatic ROI are shared by the print, not individual specimen temperatures.',
            'Validate ROI alignment and camera calibration/emissivity for this tensile geometry.',
            'Mechanical properties must be added from the tensile test; they cannot be inferred from G-code.',
        ],
    }
    def save_meta():
        (run / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    save_meta()
    with (run / 'specimens.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['specimen_id', 'condition', 'replicate', 'plate_position', 'print_order', 'gcode_object_id', 'material_batch',
                         'orientation', 'gauge_width_mm', 'gauge_thickness_mm', 'gauge_length_mm',
                         'test_date', 'test_speed_mm_min', 'max_force_n', 'uts_mpa', 'modulus_mpa',
                         'strain_at_break', 'failure_location', 'notes'])
        for order, specimen in enumerate(metadata['specimen_ids'], 1):
            writer.writerow([specimen, args.condition, replicate, '', order, '', args.material_batch, args.orientation])
    print(f'Collecting to {run.resolve()}\nRead-only: no printer settings will be changed.', flush=True)
    print('Sharpie labels (print order): ' + ', '.join(f'{i}: {label}' for i, label in enumerate(specimen_ids, 1)), flush=True)
    print(f'G-code: {selected.get("name")} | material={index.settings.get("filament_type", "unknown")} | nozzle={index.settings.get("temperature", "unknown")} C', flush=True)
    if metadata['joined_mid_print']:
        print('Joined an active print: earlier data cannot be recovered.', flush=True)
    else:
        print('Waiting for the selected print to start. Keep this terminal open.', flush=True)
    active = False
    failures = 0
    started = time.monotonic()
    last_notice = 0.0
    metadata['end_reason'] = 'interrupted_or_process_stopped'
    metadata['samples_count'] = 0
    metadata['valid_thermal_samples'] = 0
    save_meta()
    try:
        with (run / 'stream.csv').open('w', newline='') as stream, gzip.open(run / 'snapshots.jsonl.gz', 'wt', encoding='utf-8') as raw:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS, extrasaction='ignore')
            writer.writeheader()
            while True:
                tick = time.monotonic()
                row = {'timestamp_utc': now(), 'elapsed_s': round(tick - started, 3)}
                snapshot = {'timestamp_utc': row['timestamp_utc']}
                errors = []
                current = None
                try:
                    current = get_json(base + '/api/job', key)
                    failures = 0
                    snapshot['job'] = current
                    current_file = current.get('job', {}).get('file', {})
                    if (current_file.get('path'), current_file.get('origin')) != (selected['path'], selected['origin']):
                        metadata['end_reason'] = 'selected_file_changed'
                        metadata['final_job'] = current
                        break
                    state = current.get('state', '')
                    progress = current.get('progress') or {}
                    row.update(state=state, completion_percent=progress.get('completion'), filepos=progress.get('filepos'), print_time_s=progress.get('printTime'), print_time_left_s=progress.get('printTimeLeft'))
                    row.update(index.at(progress.get('filepos')))
                    if state in ('Printing', 'Paused', 'Pausing', 'Resuming', 'Starting', 'Finishing', 'Cancelling'):
                        if not active:
                            metadata['first_active_at_utc'] = row['timestamp_utc']
                            metadata['first_active_progress'] = progress
                            save_meta()
                        active = True
                    elif active and state in ('Operational', 'Ready', 'Error', 'Offline', 'Closed'):
                        metadata['end_reason'] = 'print_left_active_state'
                        metadata['final_job'] = current
                        # Preserve the last available sample below, then exit.
                except Exception as error:
                    failures += 1
                    errors.append('job: ' + str(error))
                for label, url, auth in [('printer', base + '/api/printer', key), ('thermal', args.bridge_url.rstrip('/') + '/api/thermal', '')]:
                    try:
                        payload = get_json(url, auth)
                        snapshot[label] = payload
                        if label == 'printer':
                            temps = payload.get('temperature') or {}
                            for sensor, prefix in [('tool0', 'hotend'), ('bed', 'bed')]:
                                for field in ('actual', 'target'):
                                    row[f'{prefix}_{field}_c'] = (temps.get(sensor) or {}).get(field)
                        else:
                            row.update(thermal_fields(payload, max(3, args.interval * 3)))
                            if not row['thermal_valid']:
                                errors.append('thermal: disconnected, missing or stale frame')
                    except Exception as error:
                        errors.append(label + ': ' + str(error))
                row['errors'] = '; '.join(errors)
                metadata['samples_count'] += 1
                metadata['valid_thermal_samples'] += int(bool(row.get('thermal_valid')))
                snapshot['errors'] = errors
                writer.writerow(row)
                stream.flush()
                raw.write(json.dumps(snapshot) + '\n')
                raw.flush()
                if tick - last_notice >= 10:
                    print(f'{row.get("state", "API unavailable")} | layer {row.get("gcode_layer", "?")} | ROI {row.get("roi_mean", "unavailable")} C | {row["errors"]}', flush=True)
                    last_notice = tick
                if metadata['end_reason'] == 'print_left_active_state':
                    break
                if failures >= 12:
                    metadata['end_reason'] = 'job_api_unavailable'
                    break
                time.sleep(max(0, args.interval - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        metadata['end_reason'] = 'user_stopped_collection'
    finally:
        metadata['ended_at_utc'] = now()
        save_meta()
        print(f'Saved {run.resolve()} ({metadata["end_reason"]})', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
