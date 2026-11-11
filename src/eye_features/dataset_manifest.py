"""Portable manifest inputs for videos from any dataset.

All paths are relative to the manifest's directory unless absolute. No label
meaning is inferred from filenames, folder names, KSS, or subject IDs.
"""

import csv
import json
from pathlib import Path
import re

import numpy as np


def safe_id(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", str(value)) or str(value) in {".", ".."}:
        raise ValueError(f"ID must contain letters, digits, underscore, dash or dot: {value!r}")
    return str(value)


def read_manifest(path: Path) -> list[dict]:
    """Read one row per recording; session IDs must be unique within a run."""
    records, seen = [], set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset_id", "subject_id", "session_id", "video_path"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Manifest requires columns: {', '.join(sorted(required))}")
        for row in reader:
            record = normalize_record(row, path.parent)
            if record["session_id"] in seen:
                raise ValueError("Session IDs must be unique; prefix with dataset/subject identifiers if needed")
            seen.add(record["session_id"])
            records.append(record)
    if not records:
        raise ValueError("Manifest contains no recordings")
    return records


def normalize_record(row: dict, base: Path) -> dict:
    record = dict(row)
    for key in ("dataset_id", "subject_id", "session_id"):
        record[key] = safe_id(record.get(key, ""))
    if not record.get("video_path"):
        raise ValueError("video_path is required")
    for key in ("video_path", "timestamps_path", "pvt_path"):
        value = record.get(key)
        record[key] = str((base / str(value)).resolve()) if value else None
    record["video_available"] = Path(record["video_path"]).is_file()
    record["timestamps_available"] = bool(record["timestamps_path"] and Path(record["timestamps_path"]).is_file())
    record["pvt_available"] = bool(record["pvt_path"] and Path(record["pvt_path"]).is_file())
    if record["timestamps_path"] and not record["timestamps_available"]:
        raise ValueError(f"Missing supplied timestamps: {record['timestamps_path']}")
    if record["pvt_path"] and not record["pvt_available"]:
        raise ValueError(f"Missing supplied PVT data: {record['pvt_path']}")
    raw_kss = record.get("kss")
    if raw_kss is None or str(raw_kss).strip() in {"", "0"}:
        record["kss"] = None
    else:
        score = float(raw_kss)
        if not np.isfinite(score) or score != int(score) or not 1 <= score <= 9:
            raise ValueError("KSS must be an integer 1..9, blank, or placeholder 0")
        record["kss"] = int(score)
    record["kss_label_scope"] = "session" if record["kss"] is not None else None
    record["test_id"] = record.get("test_id") or None
    record["drowsiness_label"] = record.get("drowsiness_label") or None
    record["label_scope"] = record.get("label_scope") or ("session" if record["drowsiness_label"] is not None else None)
    if record["label_scope"] not in {None, "session"}:
        raise ValueError("Manifest labels currently support session scope; timed labels need an explicit adapter")
    record["label_source"] = record.get("label_source") or None
    extras = record.get("extra_labels_json") or "{}"
    if not isinstance(json.loads(extras), dict):
        raise ValueError("extra_labels_json must be a JSON object")
    record["extra_labels_json"] = json.dumps(json.loads(extras), sort_keys=True)
    unit = record.get("timestamp_unit") or "ms"
    if unit not in {"ms", "s"}:
        raise ValueError("timestamp_unit must be ms or s")
    record["timestamp_unit"] = unit
    record["timestamp_column"] = int(record.get("timestamp_column") or 0)
    record["timestamps_skiprows"] = int(record.get("timestamps_skiprows") or 0)
    if record["timestamp_column"] < 0 or record["timestamps_skiprows"] < 0:
        raise ValueError("Timestamp column/skiprows must be nonnegative")
    fps = record.get("fps")
    record["fps"] = float(fps) if fps else None
    if record["fps"] is not None and (not np.isfinite(record["fps"]) or record["fps"] <= 0):
        raise ValueError("Explicit fps must be finite and positive")
    record["time_source"] = ("supplied_timestamps" if record["timestamps_path"] else
                             "explicit_constant_fps" if record["fps"] else "video_pts")
    record["processing_status"] = "NOT_PROCESSED" if record["video_available"] else "MISSING_RECORDING"
    return record


def load_recording_timestamps(record: dict):
    if not record["timestamps_path"]:
        return None
    path = Path(record["timestamps_path"])
    delimiter = record.get("timestamp_delimiter")
    if not delimiter:
        delimiter = "," if path.suffix.lower() == ".csv" else None
    if delimiter == "whitespace":
        delimiter = None
    values = np.loadtxt(path, delimiter=delimiter, skiprows=record["timestamps_skiprows"],
                        usecols=record["timestamp_column"], ndmin=1, dtype=float)
    timestamps_ms = values * (1000.0 if record["timestamp_unit"] == "s" else 1.0)
    if (len(values) < 2 or not np.all(np.isfinite(timestamps_ms))
            or timestamps_ms[0] < 0 or np.any(np.diff(timestamps_ms) <= 0)):
        raise ValueError(f"Need >=2 finite, nonnegative, increasing timestamps: {path}")
    return timestamps_ms


def load_pvt(record: dict):
    """Generic normalized event CSV, or the explicitly selected DROZY adapter."""
    if not record["pvt_path"]:
        return None
    if record.get("pvt_format") == "drozy":
        from .drozy import read_pvt
        return read_pvt(Path(record["pvt_path"]))
    with Path(record["pvt_path"]).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"stimulus_s", "response_s"} <= set(reader.fieldnames or []):
            raise ValueError("PVT CSV needs stimulus_s,response_s relative to the video timestamp origin")
        rows = []
        for item in reader:
            stimulus, response = float(item["stimulus_s"]), float(item["response_s"])
            if not np.isfinite(stimulus + response) or not 0 <= stimulus <= response:
                raise ValueError("Invalid PVT times")
            rows.append({"stimulus_s": stimulus, "response_s": response,
                         "reaction_time_ms": (response - stimulus) * 1000})
        return rows
