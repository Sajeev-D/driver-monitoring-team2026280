# DROZY adapter for the shared offline processor

For other datasets and the common manifest contract, see [DATASET_PROCESSING.md](DATASET_PROCESSING.md).

Run the existing MediaPipe eye extractor on prerecorded DROZY near-infrared
videos, retaining the dataset's timestamps and session labels. This adds an
offline workflow; the webcam scripts and their calibration are unchanged.

## Run on macOS

From the repository root, with the existing Python 3.12 environment:

```bash
source .venv/bin/activate
python scripts/process_drozy.py \
  --dataset /Users/sally/Downloads/DROZY \
  --sessions 1-1 1-3 \
  --output recordings/drozy_pilot
```

The defaults select subject 1, tests 1 and 3 (KSS 3 and 7). Both are processed
in full. For an initial runtime check, use `--max-seconds 70` and a separate
output directory; the manifest will mark these recordings `PARTIAL_DEBUG`.
All processing is local, with no webcam access or network service required.

After reviewing the pilot:

```bash
python scripts/process_drozy.py \
  --dataset /Users/sally/Downloads/DROZY \
  --all --output recordings/drozy_all \
  --calibration-dir calibration/drozy
```

The batch is sequential, checkpoints aggregate tables after each session, and
returns a nonzero exit code if a session fails. On a rerun with the same output
directory, matching extraction caches are reused. Failed/truncated video
decodes are rejected when their decoded frame counts differ from timestamps;
the code does not silently shift the alignment. An interrupted session restarts
its extraction, while completed session caches remain reusable.

Windows users can use `.venv\Scripts\python.exe` with the same script/flags and
their own dataset path. Do not copy the Mac virtual environment to Windows.

## Output files

| File | Contents |
| --- | --- |
| `sessions.csv` | All 42 subject/test combinations, original KSS scores, file availability, selected-session processing results. The 36 recordings present are discoverable; missing tests remain explicit. |
| `frames/SUBJECT-TEST.csv` | One row per decoded frame: elapsed milliseconds/seconds, per-eye EAR, validity, closure state when calibrated, 12 eye landmarks in pixel coordinates, approximate head pose and session KSS. |
| `windows.csv` | Main table: full 60-second windows every 10 seconds, EAR statistics, per-eye/combined PERCLOS, bilateral and per-eye closure-event features, tracking quality, approximate head pose statistics, KSS and PVT summaries. |
| `events.csv` | Calibrated continuous-recording closure events, durations, censoring and event classification. Without calibration this has headers and no events. |
| `pvt_events.csv` | Original stimulus/response times relative to each PVT origin and derived reaction times in milliseconds. |
| `calibration.csv` | Per-eye reference values, calibration status/method and profile hash for processed sessions. |
| `cache/SUBJECT-TEST.npz` | MediaPipe extraction arrays and source/model/runtime fingerprint. Reused independently of metric thresholds or calibration. |
| `qa/SUBJECT-TEST/` | EAR timeline PNG, reference-candidate montage, unannotated candidate PNGs and their frame indices/EAR values. |
| `run.json` | Selected sessions, pipeline version, source fingerprints, complete calculation settings, calibration provenance and failures. |

CSV numeric nulls are **empty fields**, never zeros. `face_found` is a binary
detection result, not a measured MediaPipe confidence. Finite EAR and face
detection establish numeric availability, not validated landmark accuracy.
Review overlays/crops before interpreting the features.

The supplied 68-point annotations, interpolation files and PSG signals are
recorded as available source files but are not used as MediaPipe outputs or
converted into physiological features. This version exports KSS and PVT labels
and eye/head metrics; it does not claim EEG, EOG or ECG feature extraction.

## Calibration

Without subject-specific reviewed references, EAR/quality/PVT/KSS exports work,
but P80, closure and blink metrics stay blank with `NO_CALIBRATION`. Missing
calibration is distinct from `LOW_VALID_DATA` in the tracking-quality column.
The webcam user's profile is never reused for dataset subjects.

Each `qa` directory contains low- and high-EAR examples chosen at distribution
quantiles. These are **inspection candidates**, not verified eyelid states.
Low EAR can result from poor tracking or pose. KSS is never used to choose
thresholds, and a drowsy face is not automatically a closed-eye reference.

After inspecting the raw candidates and nearby video frames, use the helper:

```bash
python scripts/calibrate_dataset.py \
  --cache-dir recordings/drozy_pilot/cache \
  --subject 1 \
  --open-session 1-1 --open-frame OPEN_FRAME_INDEX \
  --closed-session 1-3 --closed-frame CLOSED_FRAME_INDEX \
  --reviewer YOUR_NAME --confirm-eyes-open-and-closed \
  --output calibration/drozy/subject_1.json
```

Replace frame placeholders with **zero-based** indices from inspected examples.
Only confirm if both eyes in the open example are clearly open and both in the
closed example are suitably closed. Prefer multiple inspected frames when
assessing stability; a single-frame profile is a pilot reference, not a
validated calibration protocol. If suitable frames cannot be identified,
leave P80 unavailable. No percentile or assumed-zero closed reference is
silently substituted. The helper records source session/frame/timestamp,
per-eye EAR, cache identity, reviewer and UTC date. It rejects mismatched
subjects, missing/invalid eye tracking and EAR separation below 0.02.

Rerun the processor with `--calibration-dir calibration/drozy`; it will recompute
features from caches without rerunning MediaPipe. The resulting CSVs record
the profile hash and P80 EAR thresholds.

## Calculation and label conventions

- The final timestamp column in DROZY is elapsed **milliseconds** from the PVT
  origin. Convert it to seconds; do not use MP4 playback rate as elapsed time.
  The dataset contains roughly 15 and 30 FPS recordings, dropped frames and
  missing sessions. The processor does not synthesize frames using
  `interpIndices`, interpolate across failures, or count unobserved time as
  open/closed.
- Windows are `[start,end)`, anchored at PVT zero. Only windows with end at or
  before the last observed timestamp are exported. Incomplete start-up/tail
  windows are omitted; no partial minute is presented as a full one. Initial
  unobserved time before the first frame reduces valid-data coverage.
- Inter-frame intervals require valid eye samples at both ends and a gap no
  larger than 2.5 times the recording's median frame period. Intervals are
  clipped to window boundaries; the state at their left sample determines
  closed time. No extrapolation is made after the final sample.
- Per-eye PERCLOS is closed valid time / valid observed time (fraction 0..1).
  Combined PERCLOS is the arithmetic mean only when **both** eye values exist.
  Separate per-eye coverage and eye-disagreement duration are exported.
  `perclos_combined_percent` is 0..100. Bilateral event detection requires both
  eyes closed together. Low-coverage estimates retain their values but carry
  `LOW_VALID_DATA`; filter them before training or reporting.
- Closure events are detected once across each whole recording with the
  existing PERCLOS engine. Counts are assigned by observed termination in a
  window, excluding left/right censored events. This avoids reclassifying
  clipped long events as blinks. An event can occur in several overlapping
  windows; do not sum those counts as a dataset total. Use `events.csv` for
  unique events. Events continuing beyond a window are not counted before
  termination, so window features do not use future event duration.
- Existing engine defaults are preserved: P80 >=80% closure; candidate blink
  duration 0.15..0.40 seconds inclusive; prolonged closure **>0.40 seconds**;
  minimum valid fraction 0.80. This differs from the older context document's
  1-second prolonged threshold. Full settings are saved in `run.json`.
  Durations follow the engine's last-closed minus first-closed timestamp plus
  nominal frame period convention. Candidate blinks are not validated full
  blink ground truth. No detected blinks gives an empty mean duration, not 0 ms.
- KSS is the original ordinal session-level score, repeated with
  `kss_label_scope=session`. The dataset's placeholder 0 for test 7-1 becomes
  null. No frame/window sleepiness labels or binary drowsiness labels are
  invented from it.
- PVT summaries require stimuli in the window and responses observed before
  its end. Pending responses are not included early. Raw events are retained.
  The configurable `--pvt-lapse-ms` default of 500 ms is an explicit analysis
  convention; lapse count is not an official binary drowsiness label. No
  responses gives null reaction-time summaries and response count 0.

For model evaluation, split by **subject**, keeping sessions and overlapping
windows together. A calibration derived from an entire recording is an offline
descriptive reference; it is not a prospective deployment protocol. Fit any
learned calibration/normalization only on training or explicitly permitted
enrolment data, never on held-out labels or future test observations.

## Verification and troubleshooting

```bash
python -m pytest -q
```

Tests cover timestamp units, fractional boundary intervals, dropped frames,
missing eyes, independent per-eye results, uncalibrated exports, continuous
event boundaries/censoring, KSS placeholders and PVT response availability.

If MediaPipe reports `Could not create an NSOpenGLPixelFormat`, macOS graphics
initialization is unavailable in that execution context. Run in a local Mac
desktop terminal; within Codex a restricted sandbox may need an approved run
outside it. This does not require changing the extractor or enabling a webcam.
No extra packages are needed beyond the existing runtime dependencies.

Cache fingerprints include source timestamp/model hashes, video size/mtime,
extractor source hash and runtime versions. Increment `PIPELINE_VERSION` after
changes to extraction semantics outside `extractor.py`, such as landmark
indices, to prevent stale cache reuse. Use a fresh output directory to retain
separate experiments; aggregate CSVs are replaced on each invocation and
describe only that invocation's selected sessions.
