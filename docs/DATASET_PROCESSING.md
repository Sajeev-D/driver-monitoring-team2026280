# One video-to-CSV pipeline for different datasets

Use the same MediaPipe extractor, eye calibration, PERCLOS engine and output
schema for DROZY, other public video datasets, and your own recordings.
Dataset readers only describe sources, timestamps and labels. They do not
change the metric definitions.

`scripts/process_dataset.py` is the general entry point.
`scripts/process_drozy.py` discovers the native DROZY files and supplies that
same entry point's recording contract. The webcam demonstration is unchanged.

## Today's DROZY pilot

From the repository root with the existing Python environment:

```bash
python scripts/process_drozy.py \
  --dataset /Users/sally/Downloads/DROZY \
  --sessions 1-1 1-3 \
  --output recordings/drozy_pilot \
  --calibration-dir recordings/drozy_pilot/calibration
```

The supplied pilot profile is only for DROZY subject 1. It records an open and
a closed video frame visually inspected by Codex, plus the source indices,
EAR values and extraction fingerprints. This is a pilot reference calibration,
not experimentally validated P80 accuracy or a deployment calibration protocol.
The original KSS scores are session labels; no binary labels are invented.

The DROZY adapter also writes `input_manifest.csv`. You can run the general
processor against that manifest and get the same metrics and output columns:

```bash
python scripts/process_dataset.py \
  --manifest recordings/drozy_pilot/input_manifest.csv \
  --sessions 1-1 1-3 \
  --output recordings/drozy_pilot \
  --calibration-dir recordings/drozy_pilot/calibration
```

Both commands reuse matching extraction caches, so changing calibration or
window settings does not require another MediaPipe pass.

## Inputs for another dataset

Create a UTF-8 manifest CSV with **one row per video recording**. Required:

| Column | Meaning |
| --- | --- |
| `dataset_id` | Dataset name or version, e.g. `OUR_DATA` |
| `subject_id` | Person ID used for calibration and subject-level evaluation |
| `session_id` | Unique recording ID; include subject/clip ID if necessary |
| `video_path` | Absolute path, or relative to the manifest's folder |

IDs may contain letters, digits, underscores, dots and dashes. Session IDs
must be unique within a run, including when combining datasets. Use a separate
output directory for each dataset/experiment to keep caches and profiles clear.

Optional columns:

| Column | Default and meaning |
| --- | --- |
| `timestamps_path` | Blank: use video presentation timestamps. Otherwise a numeric elapsed-time file with one row per decoded video frame. |
| `timestamp_unit` | `ms`; also accepts `s` |
| `timestamp_column` | `0`; zero-based column index in the timestamp file |
| `timestamps_skiprows` | `0`; set to `1` for one header line |
| `timestamp_delimiter` | Comma for `.csv`, whitespace otherwise; explicit `whitespace`, `,` or `;` are supported |
| `fps` | Blank. Without a timestamp file, an explicit value generates a constant-rate timeline. Use only for recordings known to have that sampling model. |
| `kss` | Blank; original integer KSS score 1..9 if available. Zero placeholder becomes missing. |
| `drowsiness_label` | Blank; preserve the dataset's original categorical label, e.g. an annotation supplied for a clip |
| `label_scope` | `session` when a categorical label is supplied; this version accepts session/video labels only |
| `label_source` | Annotation filename, annotator or protocol identifying the label source |
| `extra_labels_json` | `{}`; JSON object for additional labels/metadata without adding dataset-specific feature columns |
| `test_id` | Optional within-subject test identifier |
| `pvt_path` | Optional event CSV with `stimulus_s,response_s` relative to the same time origin as the video timestamps |

The input contract covers video plus session/clip labels, which can represent
other datasets and your own recordings. There is no built-in reader assuming
the layout or label coding of UTARLPD: map its actual annotations into the
manifest. A dataset with time-varying labels requires a specific timed-label
adapter; do not silently propagate one label over changing ground truth.

For example, save this beside a `videos/` folder:

```csv
dataset_id,subject_id,session_id,video_path,drowsiness_label,label_scope,label_source
OUR_DATA,driver_A,driver_A_trip_01,videos/trip_01.mp4,awake,session,manual_review
OUR_DATA,driver_B,driver_B_trip_01,videos/trip_02.mp4,,,unlabelled
```

Then run:

```bash
python scripts/process_dataset.py \
  --manifest path/to/manifest.csv \
  --output recordings/our_dataset \
  --calibration-dir calibration/our_dataset
```

For supplied timing, add `timestamps_path,timestamp_unit,timestamp_column`.
An example row could use `timing/trip_01.csv,s,1`, with
`timestamps_skiprows=1` if the numeric time file has a header. Timestamps must
be finite, nonnegative, strictly increasing and share an elapsed-time origin
with any event annotations. Absolute clock timestamps must be converted to an
appropriate recording origin before use. No fixed frame rate is assumed when
timestamps are available, and frame/timestamp count mismatches fail explicitly.

Without an external file, decoded presentation timestamps are used and
validated. If the video cannot supply increasing timestamps, supply a timing
file or an explicit `fps` in the manifest. The processor never silently falls
back to a guessed 30 FPS. `time_source` records the choice in each feature row.
For recordings without an external acquisition timeline, unrecorded original
frames cannot be reconstructed or reliably identified from an encoded video.

## Shared outputs and missing fields

Every dataset gets the same **82-column `windows.csv` schema**, including when
there are no complete windows. Each file in `frames/` also uses a common schema.
Columns cover:

- Dataset/subject/session identity, original KSS/categorical labels, label scope
  and source, optional extra-label JSON and timestamp source.
- Window start/end and length, frame count, observed/nominal FPS, tracking
  coverage and missing time.
- Per-eye EAR mean, standard deviation, minimum, maximum and median.
- Per-eye PERCLOS fraction, combined fraction/percent, eye disagreement,
  open/closed references and P80 EAR thresholds.
- Per-eye and bilateral candidate blink counts/mean durations, prolonged
  closures and censored-event counts.
- Approximate head-pose mean and standard deviation for each axis.
- PVT availability, stimulus/response counts, reaction-time summaries and an
  explicitly configured lapse threshold/count.

**Unavailable values are blank.** A dataset without KSS does not get KSS 0.
Without PVT, even PVT counts are blank and `pvt_available=False`. Missing or
unverified calibration leaves P80 and event metrics unavailable, while EAR and
tracking metrics remain usable. `tracking_quality` and `calibration_status`
remain separate. Session labels retain their original meanings across datasets;
matching column names do not imply equivalent annotation protocols.

`events.csv`, `calibration.csv` and `pvt_events.csv` use fixed feature schemas.
`sessions.csv` is an input/processing inventory and may also retain
dataset-specific source-availability fields. `run.json` records the inputs,
schema version, thresholds, provenance and failures. The extraction cache
contains eye landmarks, head pose, timestamps and face-found flags, not video.
Local QA PNGs contain selected source frames and enlarged eye crops for audit.

## Calibration and metric definitions

Use `scripts/calibrate_dataset.py` on reviewed cached frames, with the subject
ID from the manifest and any session IDs belonging to that subject. It records
dataset and subject identity, preventing a profile for a different dataset
from being silently reused. Keep profiles under separate dataset directories.
See [DROZY.md](DROZY.md#calibration) for the command and visual-review workflow.

The calculation conventions are shared: time-weighted 60-second windows,
10-second stride, no missing-frame interpolation, mean of both available
per-eye PERCLOS values, bilateral events requiring both eyes closed, and
continuous event detection with censoring. See
[DROZY.md](DROZY.md#calculation-and-label-conventions) for the exact engine
defaults and event-boundary rules. `--window-s`, `--stride-s`,
`--pvt-lapse-ms`, `--sessions` and `--max-seconds` are accepted by both
entry points. Only complete windows are exported; recordings shorter than the
window length have frame/QA outputs and an empty window table with full headers.

Run `python -m pytest -q` for the shared tests. Real runtime validation today
uses DROZY; support for alternative manifest inputs is covered by synthetic
tests rather than a claim that another public dataset has been processed.
