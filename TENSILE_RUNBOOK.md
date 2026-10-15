# Tensile specimen collection

Use `tensile_logger.py` for passive collection. It does not send G-code, apply MPC,
change fan/speed/temperature/flow, or ask for an overhang angle. Leave the Pi camera
server and Mac dashboard bridge running. Stop the previous layer logger before
switching workflows. There is no need to stop the printer.

## Three specimens per condition, two per plate

Print specimens sequentially (complete one object before starting the next), with
at most two on a plate. Each condition still targets three specimens: print a plate
of two, then a plate of one with the same settings. Each job has its own run folder.
The replicate number in a folder name identifies the print job, not the specimen.
The specimen IDs remain unique across both jobs.

Specimen labels are short enough to write with a Sharpie: `T001`, `T002`, `T003`, etc.
The collector prints the labels in specimen print order. Write the label on the grip/end
region, away from the gauge section. All conditions use the same increasing sequence;
the condition and full metadata are looked up from the ID rather than encoded in it.
`print_runs/tensile/specimen_index.csv` maps labels to condition, job and print order.
Keep `specimen_registry.sqlite3` with the runs when backing up or moving the project;
it prevents reuse of reserved IDs even for interrupted runs. A separate runs root has
its own numbering sequence. Existing long IDs in older runs are left unchanged.

Record plate position, sequential print order and the G-code object ID in
`specimens.csv`. Order is prefilled 1, 2 for a two-specimen job; confirm the actual
slicer order and map the labels to the physical specimens. Sequential specimens share
bed/environment history; the second starts after the first has printed. Keep locations
and order consistent across conditions and retain the job grouping in analysis.

Begin with the Prusa ABS baseline, then vary one factor at a time: speed, cooling,
nozzle temperature, or extrusion multiplier. Hold all other settings fixed. The proposed
initial levels are in `tensile_experiment_plan.json`. With k nonbaseline levels for each
factor, the series has 1 + 4k conditions, 2(1 + 4k) print jobs and 3(1 + 4k) specimens.
Consider a final repeated baseline to check drift and counterbalance condition order
when practical.

### Proposed initial screening matrix

| Factor | Baseline | Two alternative levels | Condition IDs |
| --- | --- | --- | --- |
| Extrusion speed profile | 100% | 60%, 80% | `speed60`, `speed80` |
| Normal part cooling | 10% after first 4 layers | 0%, 30% | `fan0`, `fan30` |
| Nozzle after first layer | 260 C | 250 C, 255 C | `nozzle250`, `nozzle255` |
| Slicer extrusion multiplier | 1.00 | 0.95, 1.05 | `flow095`, `flow105` |

These are proposed screening levels, not calibrated optimal settings. They make nine
conditions / 27 specimens across 18 print jobs (2+1 per condition). A complete final
baseline repeat adds two jobs / three specimens, for 20 jobs / 30 specimens. Prusa's general ABS guidance recommends
[255 C](https://help.prusa3d.com/article/abs_2058); retain the inspected 260 C profile as
the baseline, and use proposed temperature levels only within the spool manufacturer's
range. Speed and flow steps are experimental design choices, not manufacturer-prescribed
levels. Prusa explains that [volumetric and cooling limits can reduce requested speeds](https://help.prusa3d.com/article/max-volumetric-speed_127176).

For speed, scale all non-first-layer extrusion feature speeds together (perimeters,
external/small perimeters, infill, solid/top infill, gap fill and bridges as applicable).
Keep first-layer/travel speeds, acceleration, cooling rules and volumetric limits fixed.
Verify in the sliced speed preview that the actual specimen paths differ; a lower
requested speed may have no effect where cooling or volumetric limits already dominate.
For fan, change the normal min/max cooling setpoints together while preserving bridge
fan at 20%, the four initial layers without fan and the baseline slowdown rules. This
tests normal part cooling, not a uniform fan percentage at every point in the job.
Keep first-layer nozzle temperature at 260 C for all nozzle conditions. For flow, change
only the slicer extrusion multiplier and keep firmware M221 at 100%.

The selected baseline inspected on October 15, 2026 contains:

| Setting | Value from G-code |
| --- | --- |
| Material | Generic ABS |
| Layer / nozzle size | 0.2 mm / 0.4 mm |
| Nozzle | 260 C |
| Bed | 100 C first layer, 110 C subsequent layers |
| Perimeters / infill | 2 / 15% grid |
| Perimeter / infill speeds | 170 / 200 mm/s, with feature and first-layer exceptions |
| Flow | Extrusion multiplier 1; startup M221 S100 |
| Cooling | First 4 layers disabled; normal fan 10%; bridge fan setting 20% |

This is the inspected file, not a claim about every Prusa profile. The collector archives
the actual selected file for each run. Its single G-code object label does not establish
how many physical specimens it contains: verify two specimens (or one for the second job) in the slicer preview.
Hold construction and build/loading orientation fixed; 15% infill specimens characterize
that printed construction, not automatically bulk ABS material properties.

## Before each plate

1. Slice two specimens for the first job and one for the second job with the desired condition. Use a distinct G-code filename.
   Prefer slicer settings over unrecorded manual overrides. Fan commands later in a file
   can replace a fan setting entered manually during a print.
2. Select/upload the file in OctoPrint. Verify the layout and condition in the slicer.
3. Check the thermal view: the small automatic ROI inherited from the previous project
   must actually cover deposited material for this geometry. Full sampled thermal matrices
   are saved so the ROI can be reprocessed. One ROI cannot separately measure both
   specimens; correlate it with object/position data and validate camera alignment.
4. Start the collector on the Mac before starting the print in OctoPrint. If a print is
   already running, collection can join it, but missing earlier data is unrecoverable.

From this project folder, for the current temporary Ethernet relay:

```bash
python3 tensile_logger.py --condition baseline --specimens-per-print 2 --octoprint-url http://127.0.0.1:8080
```

After the two-specimen job, select the one-specimen G-code and run:

```bash
python3 tensile_logger.py --condition baseline --specimens-per-print 1 --octoprint-url http://127.0.0.1:8080
```

For other conditions, use the same condition ID for both jobs. The collector defaults
to two specimens; explicitly pass `--specimens-per-print 1` for each single-specimen job.
The plan's target of three per condition is a planning record, not automatic printer control.

The collector waits for the selected job, records through completion/pause/cancellation,
and exits when the job leaves its active states. Leave the terminal open. Ctrl+C stops
collection only. Run the command again for the next plate; folders are never overwritten.
The relay at port 8080 is temporary. If it stops, restore the relay or use a reachable
OctoPrint URL. The collector reuses `api_key.py` or `PRUSASAM_OCTOPRINT_API_KEY`.

Inspect without collecting:

```bash
python3 tensile_logger.py --condition baseline --octoprint-url http://127.0.0.1:8080 --check-only
```

For another condition, use a descriptive condition ID and record the varied factor and
its intended level. For example, **only if the newly selected G-code is actually sliced
for 30% fan**, use:

```bash
python3 tensile_logger.py --condition fan30 --factor fan --level '30 percent' --specimens-per-print 2 --octoprint-url http://127.0.0.1:8080
```

These flags label the experiment; they do not apply or enforce its settings. Use
`--factor speed`, `--factor nozzle`, or `--factor flow` for the other series, with units in
`--level`. For flow, specify whether the level is a slicer extrusion multiplier or a
firmware M221 percentage. The former is represented in extrusion distances and slicer
metadata, and does not require an M221 change. Record spool/batch and orientation with
`--material-batch`, `--orientation`, and other context with `--notes`.

For any condition ID in the saved experiment plan, `--factor` and `--level` are filled
from the plan automatically. For example, after selecting the appropriately sliced file:

```bash
python3 tensile_logger.py --condition flow095 --octoprint-url http://127.0.0.1:8080
```

The complete proposed plan is copied into each run's metadata. The selected G-code
remains the evidence of what was sliced; a condition name does not verify its settings.

## Saved data

Each `print_runs/tensile/<condition>_rNN/` contains:

- `source.gcode`: exact downloaded file; its SHA-256 fingerprint is in metadata.
- `metadata.json`: slicer settings, factor/level, plate replicate, specimen IDs and declared sequential print order,
  initial/final job information, partial-print flag, sampling settings and termination reason.
- `stream.csv`: approximately 1 Hz temperature readings, thermal ROI statistics, print
  progress, global layer-comment count, layer Z and object/feature and file-position-derived commands; API gaps and stale
  thermal frames are flagged. Invalid thermal statistics remain blank.
- `snapshots.jsonl.gz`: sampled raw API replies, including the full thermal matrix and ROI
  coordinates. This is sampled acquisition, not every frame from the 15 Hz camera.
- `specimens.csv`: one record per declared specimen for plate position, material batch, orientation,
  measured gauge dimensions, tensile testing conditions and mechanical results.

G-code file progress may lead physical execution because of buffering. Feedrate is the
last commanded move feedrate, including travel/retractions, not measured deposition
speed. Commanded fan percentage is not measured fan RPM. Manual overrides and firmware
limits are not reconstructed. Measured nozzle/bed temperatures and their live targets
are logged separately. No layer-progress plugin is required for this collector.

After the collector exits, fill in missing experimental context in `metadata.json` and
`specimens.csv`: ambient/enclosure conditions, material drying/storage, camera calibration
and emissivity, conditioning before testing, and specimen orientation. Label the physical
specimens with the saved IDs. Keep test direction, gauge dimensions, test speed, fixture
method and conditioning comparable. Later link force/displacement or stress/strain raw
files to each ID and record the derived results with units; mechanical strength cannot
be obtained from G-code. Store mechanical raw files alongside the run.

The sampler marks acquisition termination, not successful specimen manufacture. Review
OctoPrint state, missing-data flags, physical print quality and thermal alignment before
including a specimen in analysis. An interrupted print is not a completed replicate.

## Verification

```bash
python3 -m unittest test_tensile_logger.py -v
```

For sequential prints the layer-comment count continues across objects; do not interpret
it as the layer number within a specimen. Use object ID and layer Z together. Thermal
data is attributable to a specimen only after validating the camera view and object-ID mapping.
