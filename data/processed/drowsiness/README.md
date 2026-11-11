# Drowsiness model-training tables

This folder contains the derived **window-level feature datasets** for the team.
Raw videos, face images, per-frame landmarks, PSG recordings, machine-specific
paths and large extraction caches are not included. The CSVs can be loaded
directly for model training without access to Sally's external drive.

| File | Contents |
| --- | --- |
| `drozy.csv` | 1,946 windows; 36 recordings; 14 participants |
| `uta_rldd.csv` | 8,218 windows; all 145 local video files; 48 participants |
| `combined.csv` | 10,164 windows × 273 columns; both datasets, no duplicated windows |
| `schema.json` | Exact ordered list of 273 column names |
| `camera_features.json` | Nine common camera feature names matching the synthetic-model input |
| `manifest.json` | Counts, class distribution, missingness, checksums, corrections and source coverage |
| `validation.json` | Final validation results and example eligible training-row count |
| `SOURCES.md` | Dataset citations, source links and sharing terms |

Each row describes a complete 60-second window; starts are spaced 10 seconds
apart, so consecutive rows overlap. The per-dataset CSVs and combined CSV
contain the same observations: **use one layout, do not concatenate all three**.

## Labels and missing values

- DROZY `kss`/`kss_rating` are the original session KSS score. Binary
  `drowsiness_label` is 1 for KSS >=7 and 0 otherwise.
- UTA has three video-level classes (0 alert, 5 low vigilance, 10 drowsy),
  preserved in `extra_labels_json` with fold and KSS range. Exact KSS is blank.
  Binary labels are 0 for alert and 1 for drowsy; low-vigilance binary labels
  stay blank because their KSS 6–7 range crosses the DROZY threshold.
- Blanks mean unavailable/undefined. UTA does not provide PSG/PVT. No-blink
  windows have undefined mean blink duration; fewer than two blinks means
  undefined duration variability. These must not silently become observed zero.
- `yawning_rate_per_min` is a heuristic mouth-opening candidate rate. Head-nod
  candidates and approximate head angles are also prototype features.
- Min/max EAR calibration uses the first 60 seconds of the first available clip
  per subject; each eye maps minimum EAR to openness 0 and maximum to 1.
  `calibration_overlap=True` windows overlap that prefix and should be excluded
  for post-enrollment evaluation. Extrema can be affected by tracking errors.
- Target labels are session/video-level, not independently scored per minute.

## Minimal binary training input

Example using pandas and scikit-learn in the model-training environment:

```python
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

data_dir = Path("data/processed/drowsiness")
df = pd.read_csv(data_dir / "combined.csv")
features = json.loads((data_dir / "camera_features.json").read_text())

# Example conservative policy; document changes in your experiment.
df = df.loc[
    df["drowsiness_label"].notna()
    & ~df["calibration_overlap"]
    & df["valid_data_fraction"].ge(0.8)
].dropna(subset=features).copy()

# IDs overlap across databases, so dataset_id is part of the grouping key.
groups = df["dataset_id"].astype(str) + ":" + df["subject_id"].astype(str)
split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(split.split(df, groups=groups))
X_train = df.iloc[train_idx][features]
y_train = df.iloc[train_idx]["drowsiness_label"].astype(int)
X_test = df.iloc[test_idx][features]
y_test = df.iloc[test_idx]["drowsiness_label"].astype(int)
```

This example deliberately drops incomplete camera-feature rows; a trained
imputation strategy may retain more data. Fit imputation/scaling/selection only
on training participants. Never split overlapping windows randomly by row.
UTA's official evaluation instead holds out an original subject fold at a
time; its fold is in `extra_labels_json`. A random combined-data split is not
that official protocol. Consider separate per-dataset and cross-dataset tests.

Do not automatically train on every numeric column. KSS, class IDs and labels
are targets; participant/session IDs, calibration metadata and timestamps are
not sensor features. Camera-only inference cannot use DROZY's PSG/PVT inputs.
Do not use KSS or PVT outcomes as model inputs when they define your target.

## Rebuild locally

From the repository root, with the existing `.venv` and MediaPipe model:

```bash
.venv/bin/python scripts/build_training_bundle.py \
  --uta-dataset /path/to/UTA/archive \
  --drozy-dataset /path/to/DROZY --workers 4
```

DROZY requires the full existing MediaPipe extraction caches under
`recordings/drozy_all/cache`. To extract those on another machine, first run
the documented DROZY processor; see `docs/drozy_training.md`. That step uses
the source dataset and appropriate per-subject calibration inputs. Teammates
using the packaged tables do not need to re-extract them.

UTA is decoded at native frame resolution with video PTS. Work-in-progress
caches and logs live under ignored `recordings/training_bundle_build/`.
Rerunning the command reuses matching completed per-video caches. Packaging
rejects incomplete/failed subject exports. Check the build's `progress.json`
and per-subject logs while it runs.

An isolated duplicate timestamp was found in subject 11's alert video at frame
15027. Both OpenCV and FFmpeg reported the same duplicate. Its time was changed
from 501947.100 ms to 501952.083 ms (midpoint of the valid neighbouring times),
without changing any source video or other timestamps. The clip's windows carry
`time_source=video_pts_isolated_duplicate_interpolated`; exact repairs are in
`manifest.json`. The explicit retry command is:

```bash
.venv/bin/python scripts/repair_rldd_duplicate_pts.py \
  --dataset /path/to/UTA/archive --subject 11 \
  --output recordings/training_bundle_build/subjects/11
```

After successful subject retries, `scripts/build_training_bundle.py` supports
`--package-only` with the same dataset arguments to validate and assemble the
completed exports without decoding the videos again.

Subject 41's alert container claims 18,840 frames; OpenCV and an independent
FFmpeg decode both confirm 18,839 decodable frames with no reported FFmpeg
errors. All decoded frames are processed. This container-count correction is
recorded in `manifest.json`, and the clip's windows use
`time_source=video_pts_confirmed_container_count_correction`. To reproduce that
explicitly verified retry, use the same command with `--subject 41` and
`--allow-confirmed-frame-count-correction`.

Source coverage is recorded in `manifest.json`. The initial local UTA download
contained subjects 1–48/folds 1–4, not the published full 60-subject dataset.
Subject 32's class-10 clip has two segments; windows do not cross segment files.
If additional folds are downloaded, rerun to include them.

Full feature definitions are included here: [DROZY](definitions/drozy.md) and
[UTA-RLDD](definitions/uta_rldd.md). The corresponding exporter documentation
also lives in the repository’s `docs/` folder.
