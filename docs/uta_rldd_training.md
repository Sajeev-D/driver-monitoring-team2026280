# UTA-RLDD export matching the DROZY training CSV

Run from the repository root with the existing Python environment/model:

```bash
.venv/bin/python scripts/export_rldd_training.py \
  --dataset /Volumes/SALLYII/dataset/archive
```

Writes `recordings/uta_rldd_training/uta_rldd_indicators.csv`. Its **273 column
names and order exactly match** the DROZY indicators CSV, defined in the
versioned `src/eye_features/training_schema.json`. DROZY now uses that same
fixed schema. No existing source videos or DROZY CSVs are modified.

The full video pass is substantial: each frame is processed at its native
temporal resolution so blink durations are retained. There is no implicit
frame skipping or one-frame-per-second sampling. Video presentation timestamps
(PTS) are used, preserving variable frame spacing and zeroing the first PTS.
60-second complete windows are emitted every 10 seconds, as with DROZY.

## Inventory and smaller runs

```bash
# Just discover videos and report incomplete source data; no inference.
.venv/bin/python scripts/export_rldd_training.py \
  --dataset /Volumes/SALLYII/dataset/archive --inventory-only

# One subject's alert clip, first 70 seconds, with separate debug output.
.venv/bin/python scripts/export_rldd_training.py \
  --dataset /Volumes/SALLYII/dataset/archive \
  --subjects 01 --sessions uta_01_0 --max-seconds 70 \
  --output recordings/uta_rldd_smoke

# All available classes for two subjects, at full duration.
.venv/bin/python scripts/export_rldd_training.py \
  --dataset /Volumes/SALLYII/dataset/archive --subjects 01 02
```

`--subjects`/`--sessions` select a subset. The default selects every discovered
video. `--max-seconds` is for smoke tests only, is recorded as `PARTIAL_DEBUG`
in the audit, and must cover calibration and a full window. Do not mix debug
and final CSVs. `--calibration-s`, `--window-s`, `--stride-s` are configurable.
Changing window parameters changes comparability with the default DROZY data.

## Labels: no invented KSS

The [official UTA-RLDD description](https://sites.google.com/view/utarldd/home)
defines filename labels 0=alert, 5=low vigilance, 10=drowsy. These are class
identifiers, **not exact KSS scores**. The associated KSS ranges are 1–3,
6–7 and 8–9. Exact `kss`, `kss_rating`, and `baseline_kss` stay blank.

Original class ID/name, KSS range, fold and segment are retained per row in
`extra_labels_json`, e.g.:

```json
{"fold": 1, "kss_max": 7, "kss_min": 6, "segment": null, "uta_class_id": 5, "uta_class_name": "low_vigilance"}
```

To match DROZY's `KSS >= 7` binary convention, `drowsiness_label` is 0 for
alert and 1 for drowsy; it is **blank for low vigilance**, whose KSS 6–7 range
straddles that threshold. Those rows are retained and can be used for a
three-class model with `uta_class_id` as its target. Never treat these blanks
as alert or copy the native label 5 into an exact KSS column.

Labels apply to the whole clip, not independently annotated frames/windows.
`label_rule`, `label_scope` and `label_source` record this policy. Subject IDs
are preserved; `participant_id` uses a `UTA_P` prefix to avoid collisions with
DROZY. `session_order` stays blank because class filenames do not establish
chronological recording order. `test_id` retains the filename stem.

## Discovery, split recordings and incomplete downloads

Both `Fold1_part1/Fold1_part1/01/0.mov` and official numeric subject directories
are supported, with case-insensitive video extensions. Files such as
`10_1.mp4` and `10_2.mp4` are treated as separate segments of class 10. Session
IDs are `uta_32_10_part1` / `uta_32_10_part2`. Their temporal continuity is not
assumed: no window, blink or closure crosses a file boundary. Whole and split
copies of one class are rejected together to avoid duplicate observations.
Unknown class names and duplicate subject/class/segment files fail explicitly.

Fold IDs come from folder names and are retained in `extra_labels_json`.
`inventory.json` reports all discovered videos, missing subjects relative to
the published 60-subject dataset and missing classes per present subject.
Absent recordings are not synthesized. A changing/incomplete download can
produce decode failures: let copying finish before the full extraction.

## Calibration and features

Exact per-eye minimum/maximum EAR from the first 60 seconds of each subject's
first available clip define openness 0/1. Selection is fixed from the complete
discovered inventory, ordered by subject/class/segment, independently of
`--sessions`. It is ordinarily the alert clip; if absent, the first available
class is used and the source is recorded. This ordering is an enrollment
convention, not evidence of chronological recording order.

Invalid face tracking/nonfinite EAR are excluded. The minimum open/closed gap
remains 0.02. Observed extrema can reflect tracking outliers and do not establish
physiological endpoints. Profile JSONs record the exact source frames, interval
and extraction identity. Calibration overlapping a window is flagged, and
those rows should be excluded for post-enrollment evaluation. A missing/short
or unusable enrollment clip fails explicitly instead of substituting universal
EAR thresholds or fitting endpoints from the whole evaluation recording.

Eye landmarks, approximate head pose, PERCLOS, blink/closure statistics, EAR
normalization, and head-movement candidate definitions share the existing DROZY
engine. See [DROZY definitions](drozy_training.md). Mouth landmarks come from
MediaPipe instead of DROZY's native annotations. RLDD inner-lip MAR uses mean
vertical distance of pairs (82,87), (13,14), (312,317), divided by distance
(78,308), in image pixels. The geometric definition matches DROZY's MAR, but
detector differences can change distributions. MAR >=0.6 for >=1 second remains
a **mouth-opening/yawn candidate**, not a verified yawn. Each row identifies
the landmark source. Head movements are also heuristic candidates.

PSG/EEG/EOG/EMG/ECG, PVT, exact KSS, capture-frame interpolation indices,
sleep history, absolute recording timestamps, observer rating and arbitrary
video-quality scores are **unavailable**. Their columns remain blank; explicit
PSG/PVT availability flags are false. Missing capture frames cannot be inferred
from a known 30 Hz grid for RLDD, so DROZY's dropped-frame/interpolation metrics
are not fabricated. `time_source` records PTS and window times are clip-relative.
`t_drive_min` is elapsed test/video time, not measured driving time.

## Resume, failures and verification

Decoded compact caches are stored in `cache/`. Rerunning the same command
reuses them only when source path/size/mtime, model, code/library versions and
timestamp/debug options match. Interrupted incomplete extractions are not
committed as valid caches. CSV/audit files are checkpointed after each video;
the final CSV is rebuilt for the selected videos, so stale subset rows are not
silently appended. A failure is recorded per session; successful rows are
preserved and the command exits with an error if any selected video fails.
`audit.json` distinguishes a completed selection, a partial debug run and failed
sessions. Completion of a selection does not imply the complete 180-video
dataset was downloaded: check the inventory too.

Invalid/nonincreasing PTS fail explicitly. Only if constant frame rate is an
acceptable approximation, rerun with `--timestamp-mode fps`; the resulting
`time_source` records that approximation. Both extraction options and frame
counts appear in the audit.

Use participant/fold-separated evaluation (the official protocol holds out one
fold at a time). Do not select every numeric column as a feature: class/KSS
metadata and labels are targets; IDs, timestamps, folds and calibration metadata
are not drowsiness measurements. Camera-only models should exclude unavailable
physiological fields and handle undefined blink-duration values deliberately.
