"""Create exploratory tensile-print figures; no mechanical-property inference."""
from __future__ import annotations
import argparse
import csv
import gzip
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

COLORS = {'baseline': '#2466a0', 'fan5': '#c56b17'}
STYLES = ['-', '--', ':', '-.', (0, (5, 2))]


def number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return float('nan')


def load_specimens(root):
    specimens = []
    for meta_path in sorted(root.glob('*/metadata.json')):
        meta = json.loads(meta_path.read_text())
        if not meta.get('specimen_ids') or meta.get('run_status') == 'failed_print' or not meta.get('ended_at_utc'):
            continue
        with (meta_path.parent / 'stream.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        with (meta_path.parent / 'specimens.csv').open() as handle:
            records = sorted(csv.DictReader(handle), key=lambda r: int(r['print_order']))
        object_order = list(dict.fromkeys(r.get('gcode_object_id') for r in rows if r.get('gcode_object_id') not in ('', '-1', None)))
        if len(object_order) != len(records):
            raise ValueError(f'{meta_path.parent}: object count differs from specimen count; verify mapping before plotting')
        for record, inferred_object in zip(records, object_order):
            obj = record.get('gcode_object_id') or inferred_object
            group = [r for r in rows if r.get('gcode_object_id') == obj]
            if not group:
                raise ValueError(f'{record["specimen_id"]}: no samples for object {obj}')
            times = np.array([datetime.fromisoformat(r['timestamp_utc']).timestamp() for r in group])
            values = {key: np.array([number(r.get(key)) for r in group]) for key in
                      ['roi_mean', 'hotend_actual_c', 'hotend_target_c', 'bed_actual_c', 'bed_target_c', 'gcode_fan_percent', 'thermal_age_s']}
            valid = np.array([r.get('thermal_valid') == 'True' and not r.get('errors') for r in group])
            values['roi_mean'][~valid] = np.nan
            specimens.append({'id': record['specimen_id'], 'condition': meta['condition'],
                              'run': meta_path.parent, 'object_id': obj,
                              'mapping': 'recorded' if record.get('gcode_object_id') else 'inferred_from_print_order',
                              'times': times, 'minutes': (times - times[0]) / 60,
                              'rows': group, 'values': values, 'meta': meta})
    return sorted(specimens, key=lambda s: s['id'])


def rolling_median(values, window=15):
    # Trailing median only within finite data blocks; never bridge a missing sample.
    result = np.full(len(values), np.nan)
    start = 0
    for i, value in enumerate(values):
        if not math.isfinite(value):
            start = i + 1
            continue
        result[i] = np.median(values[max(start, i - window + 1):i + 1])
    return result


def make_overview(specimens, out):
    fig, axs = plt.subplots(4, 1, figsize=(11.5, 10), sharex=True, layout='constrained')
    keys = ['roi_mean', 'hotend_actual_c', 'bed_actual_c', 'gcode_fan_percent']
    labels = ['Camera ROI mean (°C)', 'Nozzle measured (°C)', 'Bed measured (°C)', 'Fan command (%)']
    for i, specimen in enumerate(specimens):
        color = COLORS.get(specimen['condition'], f'C{i}')
        style = STYLES[i % len(STYLES)]
        for ax, key in zip(axs, keys):
            values = specimen['values'][key]
            if key == 'roi_mean':
                ax.plot(specimen['minutes'], values, color=color, alpha=.12, lw=.55)
                values = rolling_median(values)
            ax.plot(specimen['minutes'], values, color=color, ls=style, lw=1.5,
                    label=f'{specimen["id"]} ({"10% baseline" if specimen["condition"] == "baseline" else "5% fan"})')
    for ax, label in zip(axs, labels):
        ax.set_ylabel(label)
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    handles, legend_labels = axs[0].get_legend_handles_labels()
    fig.suptitle('Tensile print recordings: baseline vs 5% cooling', x=.08, y=.995, ha='left', fontsize=14)
    fig.legend(handles, legend_labels, ncol=3, fontsize=9, loc='upper left', bbox_to_anchor=(.075,.965), frameon=False)
    axs[-1].set_xlabel('Minutes since each G-code object first appeared in recorded progress')
    fig.get_layout_engine().set(rect=(0,.065,1,.85))
    fig.text(.08,.018,'ROI: faint raw samples + 15-sample trailing median. Camera alignment remains unverified.\nObject timing is approximate (printer buffering); line style identifies each specimen. No tensile results yet.',fontsize=9)
    fig.savefig(out / 'print-comparison.png', dpi=180)
    fig.savefig(out / 'print-comparison.pdf')
    plt.close(fig)


def make_frames(specimens, out):
    targets = {s['id']: (s['times'][0] + s['times'][-1]) / 2 for s in specimens}
    chosen = {}
    for run in set(s['run'] for s in specimens):
        members = [s for s in specimens if s['run'] == run]
        with gzip.open(run / 'snapshots.jsonl.gz', 'rt') as handle:
            for line in handle:
                snapshot = json.loads(line)
                when = datetime.fromisoformat(snapshot['timestamp_utc']).timestamp()
                thermal = (snapshot.get('thermal') or {}).get('thermal') or {}
                if not thermal.get('values'):
                    continue
                for specimen in members:
                    label = specimen['id']
                    gap = abs(when - targets[label])
                    if label not in chosen or gap < chosen[label][0]:
                        chosen[label] = (gap, when, thermal)
    fig, axs = plt.subplots(1, len(specimens), figsize=(15, 3.8), layout='constrained', squeeze=False)
    for ax, specimen in zip(axs[0], specimens):
        _, when, thermal = chosen[specimen['id']]
        image = ax.imshow(np.array(thermal['values']), cmap='inferno', vmin=20, vmax=160, interpolation='nearest')
        roi = thermal.get('roi') or {}
        if all(k in roi for k in ('roi_col_start', 'roi_row_start', 'roi_col_end', 'roi_row_end')):
            ax.add_patch(Rectangle((roi['roi_col_start']-.5, roi['roi_row_start']-.5),
                                  roi['roi_col_end']-roi['roi_col_start'], roi['roi_row_end']-roi['roi_row_start'],
                                  fill=False, edgecolor='#5cffff', lw=1.5))
        ax.set_title(f'{specimen["id"]} · {specimen["condition"]}\nROI {number(roi.get("roi_mean")):.1f}°C',fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(image, ax=list(axs[0]), shrink=.65, label='Camera temperature (°C)')
    fig.suptitle('Thermal view near the midpoint of each specimen recording',x=.02,ha='left',fontsize=13)
    fig.get_layout_engine().set(rect=(0,.1,1,.81))
    fig.text(.02,.02,'Cyan box = saved ROI. Shared scale 20–160°C; hotter pixels saturate. These frames help check whether the ROI covers deposited plastic.',fontsize=10)
    fig.savefig(out / 'thermal-roi-review.png',dpi=180)
    plt.close(fig)


def make_summary(specimens, out):
    summary = []
    for s in specimens:
        entry = {'specimen_id': s['id'], 'condition': s['condition'], 'run_folder': s['run'].name,
                 'object_id': s['object_id'], 'mapping_method': s['mapping'],
                 'object_interval_minutes': float(s['minutes'][-1]), 'samples': len(s['rows'])}
        for key, values in s['values'].items():
            finite = values[np.isfinite(values)]
            if len(finite):
                entry[key] = {'median': float(np.median(finite)), 'min': float(np.min(finite)), 'max': float(np.max(finite))}
        summary.append(entry)
    (out / 'specimen-summary.json').write_text(json.dumps(summary, indent=2))
    notes = '''# Exploratory print-data review

These figures compare recorded print conditions, not mechanical strength. The failed
zero-fan run is excluded. Baseline: T001–T003; 5% normal cooling: T004–T005.

- Each timeline starts when its G-code object first appears in OctoPrint progress.
  This follows buffered file progress and does not precisely timestamp nozzle motion.
- Baseline object IDs are inferred from sequential first appearance and recorded print
  order; corrected fan5 IDs use the explicit specimen sheet mapping.
- The camera ROI trace is unvalidated. Full thermal images show its position at a
  representative time and should be inspected before treating it as material temperature.
- Faint ROI lines are unsmoothed; bold lines are a trailing 15-sample median. No data
  is filtered from the saved originals. Other panels show unsmoothed measurements/commands.
- T003 was printed alone at a different bed location. Two-specimen plate positions
  also differ slightly between baseline and fan5 G-code. Bed location, print order,
  enclosure history and ROI tracking can confound a cooling comparison.
- Several specimen observations share a print job; samples are not independent repeats.
- Tensile strengths, modulus and strain-to-failure require the mechanical test data.

Files: `print-comparison.png`, `print-comparison.pdf`, `thermal-roi-review.png`,
`specimen-summary.json`. Original recordings and metadata remain in the run folders.
'''
    (out / 'README.md').write_text(notes)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path, default=Path('print_runs/tensile'))
    args = parser.parse_args()
    specimens = load_specimens(args.root)
    if not specimens:
        raise SystemExit('No completed, nonfailed specimen recordings found')
    out = args.root / 'analysis'
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.titleweight': 'normal', 'pdf.fonttype': 42})
    make_overview(specimens, out)
    make_frames(specimens, out)
    summary = make_summary(specimens, out)
    print(json.dumps([{'id':s['specimen_id'],'condition':s['condition'],'samples':s['samples'],
                       'minutes':round(s['object_interval_minutes'],2),'roi_median_c':round(s['roi_mean']['median'],2)} for s in summary],indent=2))
    print('Saved figures to',out.resolve())


if __name__ == '__main__':
    main()
