# Multimodal DROZY training CSV

Generate a separate CSV without overwriting the reviewed-reference exports:

```bash
.venv/bin/python scripts/export_drozy_training.py --dataset /Users/sally/Downloads/DROZY
```

Output: `recordings/drozy_training/drozy_indicators.csv`, one complete 60-second
window every 10 seconds. `audit.json` records row/column counts, per-column
missing counts, session inventory, source cache hashes and label counts.
`calibration/` records each subject's min/max references and source frames.

## Synthetic compatibility

The first 22 columns have exactly the synthetic dataset's names/order. They
are real DROZY measurements or explicitly identified derived proxies, rather
than synthetic data. Additional columns preserve the richer feature set.

| Synthetic column | DROZY definition |
| --- | --- |
| participant_id | Subject number formatted P001, P002, etc. |
| session_order | Original test number; missing tests are not renumbered |
| start_time | PVT test origin in local recording time, timezone unspecified |
| sleep_duration, wake_time, elapsed_time_awake | Blank: unavailable in this export |
| baseline_kss | KSS of the subject's first available session; not a separate baseline survey |
| kss_rating | Original session KSS, repeated across that session's windows |
| eye_openness | Mean of per-eye mean clipped normalized EAR; 0 closed/min, 1 open/max |
| perclos | Mean per-eye P80 closed time / valid observed time; fraction 0–1 |
| blink_rate_per_min | Completed bilateral candidate blink count per wall-clock minute |
| avg_blink_duration_ms | Mean bilateral candidate blink duration; blank when no candidates |
| blink_duration_variability_ms | Population standard deviation, not variance; blank for fewer than two blinks |
| yawning_rate_per_min | **Heuristic alias** of mouth_open_candidate_rate_per_min; not verified yawns |
| head_pitch_deg, head_yaw_deg, head_roll_deg | Mean approximate MediaPipe pose angles, degrees |
| observer_rating, video_quality_score | Blank: no observer annotation or invented video-quality score |
| camera_frame_timestamp | Window start timestamp; the row is a window, not one camera frame |
| dropped_frame_indicator | Any missing/interpolated entry on the native 30 Hz interpIndices grid in this window |
| drowsiness_label | 1 if session KSS >= 7, otherwise 0; configurable analysis convention, not official binary ground truth |

The previous model's nine feature names are present. Its `dropna` operation
still removes windows without sufficient blinks. Choose a missing-data policy
using training participants only; do not silently replace undefined durations
or variability with zero. Retrain the model on real data.

## Calibration and eye features

Each subject's first available recording supplies the first 60 seconds of
calibration (`--calibration-s` changes this duration). Exact min/max EAR are
computed separately per eye, excluding missing-face, nonfinite and negative
samples. The reference range must be at least 0.02. The same subject references
are used across that subject's sessions. Recorded extrema do not prove fully
open/closed eyelids, and tracking outliers can affect them.

Per-frame openness = clip((EAR - minimum) / (maximum - minimum), 0, 1).
`eye_openness_right/left` average those frame values; `eye_openness` averages
the two eye means. `ear_*` retains raw EAR statistics and calibration endpoints.
P80 considers closure >= 0.8 (openness <= 0.2); tracking gaps are excluded from
valid-time denominators. Counts describe completed events terminating inside
the window. Candidate blink durations are 150–400 ms; prolonged closures >400 ms.

Additional eye indicators include per-eye/bilateral blink rates, duration
standard deviation, slow candidate blink rates (300–400 ms), longest completed
prolonged closure, censored-event counts, per-eye PERCLOS, disagreement time,
tracking coverage and missing time. Rates use the 60-second wall-clock window.
`delta_perclos_5min` is the change from the window ending exactly five minutes
earlier in the same session; initial windows remain blank.

## Mouth and head movement candidates

Native 68-point **automatic** DROZY annotations are used for mouth features,
with strict frame-count alignment to original timestamps. Inner-mouth aspect
ratio is (distance(61,67)+distance(62,66)+distance(63,65))/(3*distance(60,64)),
using zero-based indices. Mean, max and valid-frame fraction are retained.
Runs with MAR >= 0.6 lasting >=1 second are mouth-opening candidates; speaking
or other movements can cause false positives. `yawning_status` identifies this
proxy. These arbitrary prototype thresholds are not validated annotations.

Head mean/std/range come from the existing approximate MediaPipe pose. Absolute
pitch excursions >=10 degrees from the median of the current recording's first
60 seconds, lasting 0.3–4 seconds, are **head-movement candidates**, not verified
nods. Completed candidate events only; missing frames and large gaps censor
runs. `head_nod_status` records this limitation.

## KSS and PVT

Original KSS comes directly from the subject row/test column in `KSS.txt`.
`kss`, `kss_rating`, label scope, source and rule are included. There are no
minute-by-minute KSS observations. `t_drive_min` means minutes since test start;
DROZY is a PVT experiment and this is not measured driving time.

PVT features include stimulus/response counts, mean/median/std/90th-percentile
reaction time, lapses >=500 ms, lapse fraction and responses <100 ms (false-start
candidates). Only stimuli and responses both observed inside the window enter
RT statistics. These thresholds are analysis conventions. PVT metrics are
behavioral outcomes and may be excluded from a camera-only deployed model.

## PSG features and alignment

The continuous EDF reader follows the [official EDF specification](https://www.edfplus.info/specs/edf.html).
It verifies header/payload lengths, supports distinct channel sample rates,
and converts signed int16 samples using digital/physical min/max calibration.
Physical units are retained per channel; no unit conversion is assumed.

Available channels Fz/Cz/C3/C4/Pz/Oz, EOG-V/EOG-H, EMG and ECG provide mean,
standard deviation, RMS, peak-to-peak and saturation fraction. Oz is absent
from most recordings and stays blank. EEG channels additionally provide absolute
band power (physical-unit squared), relative power and theta/alpha and
(theta+alpha)/beta ratios. Bands: delta [0.5,4), theta [4,8), alpha [8,13),
beta [13,30) Hz. Relative power denominator is total power over [0.5,30) Hz.
Welch PSD uses mean-removed 4-second Hann segments with 50% overlap. No artifact
rejection, sleep staging, validated EOG event detection, ECG beat detection,
heart-rate or HRV inference is performed. Saturation is exposed for screening.

PSG window positions use PVT origin minus EDF start time. EDF header time has
whole-second precision: **subsecond synchronization is not established**.
`psg_alignment_status` and `psg_alignment_uncertainty_s=1` expose this limitation;
Cam-Sync/PVT trigger refinement remains future work. Partial PSG windows remain
blank and `psg_window_complete=False`. These are exploratory summaries, not
validated physiological drowsiness scores.

`interpIndices` measures missingness on the original 30 Hz PVT grid; it does
not fabricate video measurements. Raw full-frame intensity/depth images are
absent in the July 2018 release. Manual annotations/intrinsics are reference
material, not standalone drowsiness labels; they are not transformed into
invented indicators.

## Training precautions

- Split **by participant**, keeping overlapping windows and sessions together.
- Exclude `calibration_overlap=True` rows for post-calibration evaluation.
- `kss`, `kss_rating`, KSS-derived labels/rules and baseline KSS are target or
  target-related metadata; never include them by automatically selecting all
  numeric columns. Subject/session IDs and timestamps are metadata too.
- Prefer camera-only feature subsets for webcam inference. PSG/PVT are separate
  modalities and must be available at deployment to serve as model inputs.
- Screen quality, candidate definitions, saturation and missingness. Fit any
  imputation/scaling/selection on training participants only.
- First-session calibration assumes subject enrollment is available before
  evaluated recordings. It is not a validated enrollment procedure.
