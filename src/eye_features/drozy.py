"""DROZY-specific file discovery and label adapters for the common pipeline."""

from pathlib import Path
from datetime import datetime
import numpy as np

from .dataset_manifest import normalize_record
from .offline import run_records, write_csv

def read_timestamps(path: Path) -> np.ndarray:
    data = np.loadtxt(path, dtype=np.int64, ndmin=2)
    if data.shape[1] != 8 or len(data) < 2:
        raise ValueError(f"Expected >=2 rows of eight timestamp columns: {path}")
    timestamps_ms = data[:, 7]
    if timestamps_ms[0] < 0 or np.any(np.diff(timestamps_ms) <= 0):
        raise ValueError(f"Elapsed timestamps must be nonnegative and increasing: {path}")
    return timestamps_ms


def discover_sessions(root: Path) -> list[dict]:
    kss = np.loadtxt(root / "KSS.txt", dtype=int, ndmin=2)
    if kss.shape != (14, 3) or np.any((kss < 0) | (kss > 9)):
        raise ValueError("KSS.txt must be a 14-by-3 table with scores 0..9")
    rows = []
    for subject in range(1, 15):
        for test in range(1, 4):
            sid = f"{subject}-{test}"
            files = {
                "video": root / "videos_i8" / f"{sid}.mp4",
                "timestamps": root / "timestamps" / f"{sid}.txt",
                "pvt": root / "pvt-rt" / f"{sid}.csv",
                "landmarks_2d": root / "annotations-auto" / f"{sid}-s2.txt",
                "landmarks_3d": root / "annotations-auto" / f"{sid}-s3.txt",
                "psg": root / "psg" / f"{sid}.edf",
                "interpolation": root / "interpIndices" / f"{sid}.txt",
            }
            score = int(kss[subject - 1, test - 1])
            rows.append({
                "subject_id": subject, "session_id": sid, "test_id": test,
                "kss": score if score else None, "kss_label_scope": "session",
                **{f"{name}_available": path.is_file() for name, path in files.items()},
                "processing_status": "NOT_PROCESSED" if files["video"].is_file()
                                     else "MISSING_RECORDING",
            })
    return rows


def read_pvt(path: Path) -> list[dict]:
    """Return stimulus/response seconds relative to the PVT header origin."""
    fmt = "%Y-%m-%d_%H.%M.%S.%f"
    lines = path.read_text(encoding="utf-8").splitlines()
    origin = datetime.strptime(lines[0].strip(), fmt)
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        stimulus, response = (datetime.strptime(x.strip(), fmt) for x in line.split(";"))
        start, end = ((x - origin).total_seconds() for x in (stimulus, response))
        if start < 0 or end < start:
            raise ValueError(f"Invalid PVT stimulus/response order: {path}")
        rows.append({"stimulus_s": start, "response_s": end,
                     "reaction_time_ms": (end - start) * 1000.0})
    return rows


def drozy_records(root: Path):
    """Translate the native DROZY layout into the shared recording contract."""
    records = []
    for session in discover_sessions(root):
        sid = session["session_id"]
        record = normalize_record({
            **session, "dataset_id": "DROZY",
            "video_path": str(root / "videos_i8" / f"{sid}.mp4"),
            "timestamps_path": str(root / "timestamps" / f"{sid}.txt") if session["timestamps_available"] else None,
            "timestamp_column": 7, "timestamp_unit": "ms", "timestamp_delimiter": "whitespace",
            "pvt_path": str(root / "pvt-rt" / f"{sid}.csv") if session["pvt_available"] else None,
            "pvt_format": "drozy", "label_source": "DROZY/KSS.txt",
        }, root)
        if session["video_available"] and not session["timestamps_available"]:
            raise ValueError(f"DROZY recording {sid} is missing required source timestamps")
        records.append(record)
    return records


def run_batch(root: Path, output: Path, model: Path, sessions=None,
              calibration_dir=None, window_s=60.0, stride_s=10.0,
              max_seconds=None, pvt_lapse_ms=500.0):
    records = drozy_records(root.resolve())
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "input_manifest.csv", records)
    return run_records(records, output, model, sessions, calibration_dir,
                       window_s, stride_s, max_seconds, pvt_lapse_ms)
