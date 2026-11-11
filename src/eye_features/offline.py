"""Dataset-independent video extraction, eye metrics and common CSV exports."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .calibration import EyeCalibration
from .perclos import PERCLOSConfig, calculate_ear_series, calculate_perclos
from .dataset_manifest import safe_id

PIPELINE_VERSION = 2
EYE_NAMES = ("right", "left")
LABEL_FIELDS = ["dataset_id", "subject_id", "session_id", "test_id", "kss", "kss_label_scope", "drowsiness_label", "label_scope", "label_source", "extra_labels_json"]
WINDOW_FIELDS = ['dataset_id',
 'subject_id',
 'session_id',
 'test_id',
 'kss',
 'kss_label_scope',
 'drowsiness_label',
 'label_scope',
 'label_source',
 'extra_labels_json',
 'window_start_s',
 'window_end_s',
 'window_complete',
 'window_s',
 'stride_s',
 'frame_count',
 'time_source',
 'observed_fps',
 'nominal_fps',
 'face_found_fraction_frames',
 'calibration_status',
 'calibration_method',
 'calibration_profile_sha256',
 'landmark_source',
 'quality_status',
 'valid_time_s',
 'valid_data_fraction',
 'missing_time_s',
 'tracking_quality',
 'eye_openness',
 'eye_openness_right',
 'eye_openness_left',
 'ear_right_mean',
 'ear_right_std',
 'ear_right_min',
 'ear_right_max',
 'ear_right_median',
 'valid_time_right_s',
 'valid_fraction_right',
 'closed_time_right_s',
 'perclos_right',
 'ear_open_right',
 'ear_closed_right',
 'p80_ear_threshold_right',
 'ear_left_mean',
 'ear_left_std',
 'ear_left_min',
 'ear_left_max',
 'ear_left_median',
 'valid_time_left_s',
 'valid_fraction_left',
 'closed_time_left_s',
 'perclos_left',
 'ear_open_left',
 'ear_closed_left',
 'p80_ear_threshold_left',
 'perclos_combined',
 'perclos_combined_percent',
 'perclos_eye_difference',
 'eye_disagreement_time_s',
 'candidate_blink_count_right',
 'mean_blink_duration_right_ms',
 'prolonged_closure_count_right',
 'censored_event_count_right',
 'candidate_blink_count_left',
 'mean_blink_duration_left_ms',
 'prolonged_closure_count_left',
 'censored_event_count_left',
 'candidate_blink_count_both',
 'mean_blink_duration_both_ms',
 'prolonged_closure_count_both',
 'censored_event_count_both',
 'pvt_available',
 'pvt_stimulus_count',
 'pvt_response_count',
 'pvt_mean_rt_ms',
 'pvt_median_rt_ms',
 'pvt_lapse_threshold_ms',
 'pvt_lapse_count',
 'head_pitch_approx_mean_deg',
 'head_pitch_approx_std_deg',
 'head_yaw_approx_mean_deg',
 'head_yaw_approx_std_deg',
 'head_roll_approx_mean_deg',
 'head_roll_approx_std_deg']
EVENT_FIELDS = [*LABEL_FIELDS, "eye", "start_s", "last_closed_sample_s", "observed_end_s", "duration_s", "left_censored", "right_censored", "candidate_blink", "prolonged_closure"]

def label_fields(record):
    return {key: record.get(key) for key in LABEL_FIELDS}

def write_csv(path: Path, rows: list[dict], fields=None):
    """Write stable headers, blank numeric nulls, and UTF-8 CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    if not fields:
        raise ValueError(f"No CSV schema supplied for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if isinstance(v, (float, np.floating))
                                  and not np.isfinite(v) else v)
                             for k, v in row.items()})


def load_calibration(directory: Path | None, subject: str, dataset_id=None):
    """Only verified, provenance-bearing profiles may enable P80.

    Image review is a human/researcher judgement, not inferred from KSS or EAR.
    Profiles are deliberately separate from the webcam user's calibration.
    """
    if directory is None or not (directory / f"subject_{subject}.json").is_file():
        return None, {"status": "NO_CALIBRATION", "method": "none"}
    path = directory / f"subject_{subject}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Calibration profile must be a JSON object: {path}")
    if str(data.get("subject_id")) != str(subject):
        raise ValueError(f"Calibration subject mismatch: {path}")
    if dataset_id is not None and data.get("dataset_id") != dataset_id:
        raise ValueError(f"Calibration dataset mismatch: {path}")
    minmax = data.get("method") == "minmax_ear"
    if minmax and (not data.get("sources") or not data.get("calibration_interval_s")):
        raise ValueError(f"Min/max calibration requires sources and calibration interval: {path}")
    if not minmax and data.get("verified_open_and_closed") is not True:
        return None, {"status": "UNVERIFIED_REFERENCES", "method": "unverified"}
    if not minmax and (not data.get("reviewer") or not data.get("sources") or not data.get("reviewed_at")):
        raise ValueError(f"Verified calibration requires reviewer, reviewed_at and sources: {path}")
    if not isinstance(data["sources"], list) or not all(isinstance(ref, dict) for ref in data["sources"]):
        raise ValueError(f"Calibration sources must be a list of reference objects: {path}")
    try:
        calibration = EyeCalibration(tuple(data["ear_open"]), tuple(data["ear_closed"]))
    except (KeyError, TypeError) as error:
        raise ValueError(f"Calibration requires per-eye ear_open and ear_closed arrays: {path}") from error
    return calibration, {"status": "MINMAX_REFERENCES" if minmax else "VERIFIED_REFERENCES",
                         "method": "minmax_ear" if minmax else "reviewed_video_frames",
                         "profile": str(path), "profile_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "details": data}


def calibration_from_minmax(cache_dir: Path, subject: str, session: str,
                            start_s: float, end_s: float) -> dict:
    """Fit per-eye extrema from an explicitly chosen calibration interval."""
    if not np.isfinite([start_s, end_s]).all() or start_s < 0 or end_s <= start_s:
        raise ValueError("Calibration interval must satisfy 0 <= start < end.")
    safe_id(session)
    with np.load(cache_dir / f"{session}.npz", allow_pickle=False) as cache:
        identity = json.loads(str(cache["identity"]))
        if str(identity.get("subject_id")) != str(subject):
            raise ValueError("Calibration session subject mismatch")
        times = cache["timestamps_ms"] / 1000.0
        if not len(times) or start_s < times[0] or end_s > times[-1]:
            raise ValueError("Calibration interval is outside the cached recording")
        ears = ears_from_landmarks(cache["landmarks"])
        valid = (cache["face_found"].astype(bool) & (times >= start_s) & (times <= end_s)
                 & np.all(np.isfinite(ears) & (ears >= 0), axis=1))
        indices = np.flatnonzero(valid)
        calibration = EyeCalibration.from_ear_samples(ears[valid])
        sources = []
        for eye, name in enumerate(EYE_NAMES):
            for state, func in (("closed", np.argmin), ("open", np.argmax)):
                index = int(indices[func(ears[valid, eye])])
                sources.append({"eye": name, "state": state, "session_id": session,
                                "frame_index": index, "ear": float(ears[index, eye]),
                                "extraction_identity": identity})
    return {"version": 1, "subject_id": str(subject), "dataset_id": identity["dataset_id"],
            "ear_open": list(calibration.ear_open), "ear_closed": list(calibration.ear_closed),
            "method": "minmax_ear", "calibration_interval_s": [start_s, end_s],
            "valid_sample_count": len(indices), "sources": sources,
            "extraction_identity": identity,
            "limitation": "Exact observed extrema; tracking outliers can affect endpoints. "
                          "Observed extrema do not establish fully open/closed eyelids."}


def ears_from_landmarks(landmarks: np.ndarray) -> np.ndarray:
    return np.column_stack([calculate_ear_series(landmarks[:, i]) for i in range(2)])


def calibration_from_reviewed_frames(cache_dir: Path, subject: str, open_session: str,
                                     open_frame: int, closed_session: str,
                                     closed_frame: int, reviewer: str) -> dict:
    """Build a profile only after explicit visual confirmation in the CLI."""
    if not reviewer.strip():
        raise ValueError("A reviewer name is required")
    references, values = [], []
    for state, sid, index in (("open", open_session, open_frame),
                              ("closed", closed_session, closed_frame)):
        safe_id(sid)
        path = cache_dir / f"{sid}.npz"
        with np.load(path, allow_pickle=False) as cache:
            identity = json.loads(str(cache["identity"]))
            if str(identity.get("subject_id")) != str(subject):
                raise ValueError(f"Reference {sid} does not belong to subject {subject}")
            if references and identity.get("dataset_id") != references[0]["extraction_identity"].get("dataset_id"):
                raise ValueError("References must belong to the same dataset")
            if index < 0 or index >= len(cache["timestamps_ms"]):
                raise ValueError(f"Reference frame {index} is outside {sid}")
            ear = ears_from_landmarks(cache["landmarks"][index:index+1])[0]
            if not cache["face_found"][index] or not np.all(np.isfinite(ear)):
                raise ValueError(f"Reference {sid}/{index} has unusable eye tracking")
            values.append(ear.tolist())
            references.append({"state": state, "session_id": sid, "frame_index": index,
                               "timestamp_ms": float(cache["timestamps_ms"][index]),
                               "ear_right_left": ear.tolist(),
                               "extraction_identity": json.loads(str(cache["identity"]))})
    calibration = EyeCalibration(tuple(values[0]), tuple(values[1]))
    return {"version": 1, "subject_id": str(subject), "dataset_id": references[0]["extraction_identity"]["dataset_id"], "ear_open": list(calibration.ear_open),
            "ear_closed": list(calibration.ear_closed), "verified_open_and_closed": True,
            "reviewer": reviewer.strip(), "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "sources": references, "method": "reviewed_video_frames",
            "limitation": "Pilot image reference; visual confirmation is not physiological validation"}


@dataclass
class Timeline:
    """Engine classifications and supported intervals for one complete recording."""

    timestamps_s: np.ndarray
    ear: np.ndarray
    face_found: np.ndarray
    valid: np.ndarray
    interval_valid: np.ndarray
    nominal_period_s: float
    metrics: dict
    events: list[dict]


def make_timeline(landmarks, timestamps_ms, face_found, calibration=None,
                  config=None) -> Timeline:
    config = config or PERCLOSConfig()
    timestamps_s = np.asarray(timestamps_ms, dtype=float) / 1000.0
    if len(timestamps_s) < 2 or not np.all(np.isfinite(timestamps_s)) or np.any(np.diff(timestamps_s) <= 0):
        raise ValueError("Need >=2 finite, strictly increasing timestamps in milliseconds")
    ear = ears_from_landmarks(landmarks)
    face_found = np.asarray(face_found, dtype=bool)
    if face_found.shape != (len(timestamps_s),) or len(ear) != len(timestamps_s):
        raise ValueError("Landmark, face and timestamp lengths differ")
    valid = np.isfinite(ear) & face_found[:, None]
    period = float(np.median(np.diff(timestamps_s)))
    gaps = np.diff(timestamps_s) > config.max_gap_factor * period
    interval_valid = valid[:-1] & valid[1:] & ~gaps[:, None]
    metrics, events = {}, []
    if calibration is not None:
        for index, name in enumerate((*EYE_NAMES, "both")):
            bilateral = name == "both"
            metrics[name] = calculate_perclos(
                landmarks if bilateral else landmarks[:, index], timestamps_s,
                calibration.ear_open if bilateral else calibration.ear_open[index],
                calibration.ear_closed if bilateral else calibration.ear_closed[index],
                face_lost=~face_found, config=config,
            )
            metric = metrics[name]
            for event in metric["closure_events"]:
                first, last = event["start_index"], event["end_index"]
                left_censored = (first == 0 or not metric["valid_mask"][first - 1]
                                 or gaps[first - 1])
                right_censored = not event["completed"]
                complete = not left_censored and not right_censored
                duration = event["duration_s"]
                # Assign counts when the event's termination is observed. No
                # partial events become new blinks at a rolling-window edge.
                observed_end = (float(timestamps_s[last + 1])
                                if last + 1 < len(timestamps_s) else float(timestamps_s[last]))
                events.append({
                    "eye": name, "start_s": event["start_time_s"],
                    "last_closed_sample_s": event["end_time_s"],
                    "observed_end_s": observed_end, "duration_s": duration,
                    "left_censored": left_censored, "right_censored": right_censored,
                    "candidate_blink": complete and config.blink_min_s <= round(duration, 9) <= config.blink_max_s,
                    "prolonged_closure": complete and round(duration, 9) > config.prolonged_closure_min_s,
                })
    return Timeline(timestamps_s, ear, face_found, valid, interval_valid, period, metrics, events)


def window_metrics(timeline: Timeline, session: dict, pvt: list[dict],
                   calibration_metadata: dict, window_s=60.0, stride_s=10.0,
                   min_valid_fraction=0.8, pvt_lapse_ms=500.0) -> list[dict]:
    """Full [start,end) windows, anchored to PVT zero, without gap filling.

    PERCLOS integrates clipped valid inter-frame intervals (left-sample state).
    The mean of both per-eye values is only emitted when both are available.
    Event counts use observed termination in [start,end), and exclude censored
    events. They are descriptive window features, not unique dataset totals.
    """
    if not all(np.isfinite(x) and x > 0 for x in (window_s, stride_s, pvt_lapse_ms)):
        raise ValueError("Window, stride and PVT lapse threshold must be finite and >0")
    ts = timeline.timestamps_s
    rows = []
    first_start = max(0.0, float(np.floor(ts[0] / stride_s) * stride_s))
    for start in np.arange(first_start, max(first_start, ts[-1] - window_s) + 1e-9, stride_s):
        end = float(start + window_s)
        if end > ts[-1] + 1e-9:
            continue
        weights = np.maximum(0, np.minimum(ts[1:], end) - np.maximum(ts[:-1], start))
        mask = (ts >= start) & (ts < end)
        row = label_fields(session)
        row.update({
            "window_start_s": float(start), "window_end_s": end, "window_complete": True,
            "window_s": window_s, "stride_s": stride_s, "frame_count": int(mask.sum()),
            "time_source": session.get("time_source", "supplied_timestamps"),
            "observed_fps": float(mask.sum() / window_s),
            "nominal_fps": 1.0 / timeline.nominal_period_s,
            "face_found_fraction_frames": float(np.mean(timeline.face_found[mask])) if mask.any() else np.nan,
            "calibration_status": calibration_metadata["status"],
            "calibration_method": calibration_metadata["method"],
            "calibration_profile_sha256": calibration_metadata.get("profile_sha256", ""),
            "landmark_source": "mediapipe", "quality_status": "OK",
        })
        bilateral_valid = np.all(timeline.interval_valid, axis=1)
        valid_time = float(weights @ bilateral_valid)
        row["valid_time_s"] = valid_time
        row["valid_data_fraction"] = valid_time / window_s
        row["missing_time_s"] = window_s - valid_time
        row["tracking_quality"] = "OK" if valid_time / window_s >= min_valid_fraction else "LOW_VALID_DATA"
        row["quality_status"] = ("NO_CALIBRATION" if not timeline.metrics else row["tracking_quality"])
        for i, name in enumerate(EYE_NAMES):
            values = timeline.ear[mask & timeline.valid[:, i], i]
            for stat, func in (("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max),
                               ("median", np.median)):
                row[f"ear_{name}_{stat}"] = float(func(values)) if len(values) else np.nan
            vt = float(weights @ timeline.interval_valid[:, i])
            row[f"valid_time_{name}_s"] = vt
            row[f"valid_fraction_{name}"] = vt / window_s
            if timeline.metrics:
                result = timeline.metrics[name]
                closed = timeline.interval_valid[:, i] & result["closed_mask"][:-1]
                ct = float(weights @ closed)
                row[f"closed_time_{name}_s"] = ct
                row[f"perclos_{name}"] = ct / vt if vt else np.nan
                row[f"ear_open_{name}"] = result["ear_open"]
                row[f"ear_closed_{name}"] = result["ear_closed"]
                row[f"p80_ear_threshold_{name}"] = result["p80_ear_threshold"]
            else:
                for key in ("closed_time_{}_s", "perclos_{}", "ear_open_{}", "ear_closed_{}", "p80_ear_threshold_{}"):
                    row[key.format(name)] = np.nan
        for name in EYE_NAMES:
            metric = timeline.metrics.get(name)
            if metric:
                index = EYE_NAMES.index(name)
                selected = mask & timeline.valid[:, index]
                openness = 1.0 - metric["closure_fraction"][selected]
                row[f"eye_openness_{name}"] = float(np.mean(openness)) if len(openness) else np.nan
            else:
                row[f"eye_openness_{name}"] = np.nan
        row["eye_openness"] = (row["eye_openness_right"] + row["eye_openness_left"]) / 2
        r, l = row["perclos_right"], row["perclos_left"]
        row["perclos_combined"] = (r + l) / 2 if np.isfinite(r) and np.isfinite(l) else np.nan
        row["perclos_combined_percent"] = row["perclos_combined"] * 100
        row["perclos_eye_difference"] = abs(r - l)
        row["eye_disagreement_time_s"] = (
            float(weights @ (bilateral_valid & (timeline.metrics["right"]["closed_mask"][:-1]
                                                 != timeline.metrics["left"]["closed_mask"][:-1])))
            if timeline.metrics else np.nan
        )
        for name in (*EYE_NAMES, "both"):
            event_rows = [e for e in timeline.events if e["eye"] == name and start <= e["observed_end_s"] < end]
            blinks = [e for e in event_rows if e["candidate_blink"]]
            available = bool(timeline.metrics) and (valid_time > 0 if name == "both"
                        else row[f"valid_time_{name}_s"] > 0)
            row[f"candidate_blink_count_{name}"] = len(blinks) if available else np.nan
            row[f"mean_blink_duration_{name}_ms"] = float(np.mean([e["duration_s"] for e in blinks]) * 1000) if blinks else np.nan
            row[f"prolonged_closure_count_{name}"] = sum(e["prolonged_closure"] for e in event_rows) if available else np.nan
            row[f"censored_event_count_{name}"] = sum(e["left_censored"] or e["right_censored"] for e in event_rows) if available else np.nan
        stimuli = [e for e in (pvt or []) if start <= e["stimulus_s"] < end]
        responses = [e["reaction_time_ms"] for e in stimuli if e["response_s"] < end]
        row.update({
            "pvt_available": pvt is not None,
            "pvt_stimulus_count": len(stimuli) if pvt is not None else np.nan, "pvt_response_count": len(responses) if pvt is not None else np.nan,
            "pvt_mean_rt_ms": float(np.mean(responses)) if responses else np.nan,
            "pvt_median_rt_ms": float(np.median(responses)) if responses else np.nan,
            "pvt_lapse_threshold_ms": pvt_lapse_ms,
            "pvt_lapse_count": sum(rt >= pvt_lapse_ms for rt in responses) if pvt is not None else np.nan,
        })
        rows.append(row)
    return rows


def fingerprint(video: Path, timestamps: Path, model: Path) -> dict:
    """Bind caches to source files, model bytes and extractor implementation."""
    from . import extractor
    import cv2
    import mediapipe
    stat = video.stat()
    return {
        "pipeline_version": PIPELINE_VERSION, "video_path": str(video.resolve()),
        "video_size": stat.st_size, "video_mtime_ns": stat.st_mtime_ns,
        "timestamps_sha256": hashlib.sha256(timestamps.read_bytes()).hexdigest() if timestamps else None,
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "extractor_sha256": hashlib.sha256(Path(extractor.__file__).read_bytes()).hexdigest(),
        "mediapipe_version": mediapipe.__version__, "opencv_version": cv2.__version__,
        "min_face_confidence": 0.5,
    }


def extract_recording(record: dict, model: Path, output: Path, max_seconds=None):
    """Decode every frame with supplied time, video PTS, or explicit fixed FPS."""
    import cv2
    from .extractor import EyeFeatureExtractor
    from .dataset_manifest import load_recording_timestamps

    video = Path(record["video_path"])
    sid = record["session_id"]
    timestamp_path = Path(record["timestamps_path"]) if record["timestamps_path"] else None
    ts = load_recording_timestamps(record)
    identity = fingerprint(video, timestamp_path, model)
    identity.update({"max_seconds": max_seconds,
                     **{key: record.get(key) for key in ("dataset_id", "subject_id", "time_source", "fps",
                        "timestamp_unit", "timestamp_column", "timestamps_skiprows", "timestamp_delimiter")}})
    cache_path = output / "cache" / f"{sid}.npz"
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cache:
            if json.loads(str(cache["identity"])) == identity:
                print(f"{sid}: reusing validated extraction cache", flush=True)
                return {k: cache[k].copy() for k in ("landmarks", "timestamps_ms", "face_found", "head_pose")}, identity
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video}")
    lm, faces, poses, captured_ts = [], [], [], []
    try:
        with EyeFeatureExtractor(model) as extractor:
            while True:
                index = len(lm)
                if ts is not None and max_seconds is not None and index < len(ts) and ts[index] > max_seconds * 1000:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                if ts is not None:
                    if index >= len(ts):
                        raise ValueError(f"{sid}: more decoded frames than timestamp rows")
                    timestamp_ms = float(ts[index])
                elif record["fps"]:
                    timestamp_ms = index * 1000.0 / record["fps"]
                else:
                    timestamp_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                if not np.isfinite(timestamp_ms) or timestamp_ms < 0 or (captured_ts and timestamp_ms <= captured_ts[-1]):
                    raise ValueError(f"{sid}: invalid/nonincreasing video timestamps; supply timestamps_path or explicit fps")
                if max_seconds is not None and timestamp_ms > max_seconds * 1000:
                    break
                result = extractor.process(frame, round(timestamp_ms))
                lm.append(result.landmarks)
                faces.append(result.face_found)
                poses.append(result.head_pose)
                captured_ts.append(timestamp_ms)
                if index % 1500 == 0:
                    total = str(len(ts)) if ts is not None else "?"
                    print(f"{sid}: frame {index+1}/{total}, t={timestamp_ms/1000:.1f}s", flush=True)
    finally:
        cap.release()
    if ts is not None:
        expected = len(ts) if max_seconds is None else int(np.sum(ts <= max_seconds * 1000))
        if len(lm) != expected:
            raise ValueError(f"{sid}: decoded {len(lm)} frames but expected {expected}; cannot safely align timestamps")
    if len(lm) < 2:
        raise ValueError(f"{sid}: fewer than two decoded frames")
    arrays = {"landmarks": np.asarray(lm, dtype=np.float32),
              "timestamps_ms": np.asarray(captured_ts, dtype=np.float64),
              "face_found": np.asarray(faces, dtype=bool),
              "head_pose": np.asarray(poses, dtype=np.float32)}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp = cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **arrays, identity=json.dumps(identity, sort_keys=True))
    temp.replace(cache_path)
    return arrays, identity


def export_frames(path: Path, arrays: dict, timeline: Timeline, session: dict, metadata: dict):
    def rows():
        for i, ts in enumerate(arrays["timestamps_ms"]):
            row = {**label_fields(session), "time_source": session.get("time_source", "supplied_timestamps"),
                   "frame_index": i, "timestamp_ms": float(ts), "time_s": ts / 1000,
                   "face_found": bool(arrays["face_found"][i]),
                   "calibration_status": metadata["status"], "landmark_source": "mediapipe"}
            for eye, name in enumerate(EYE_NAMES):
                row[f"ear_{name}"] = float(timeline.ear[i, eye])
                row[f"valid_{name}"] = bool(timeline.valid[i, eye])
                metric = timeline.metrics.get(name)
                valid = timeline.valid[i, eye]
                row[f"closure_fraction_{name}"] = float(metric["closure_fraction"][i]) if metric and valid else np.nan
                row[f"eye_openness_{name}"] = 1.0 - row[f"closure_fraction_{name}"]
                row[f"p80_closed_{name}"] = bool(metric["closed_mask"][i]) if metric and valid else None
                for point in range(6):
                    for axis, axis_name in enumerate(("x", "y")):
                        row[f"{name}_p{point+1}_{axis_name}_px"] = float(arrays["landmarks"][i, eye, point, axis])
            for j, axis in enumerate(("pitch", "yaw", "roll")):
                row[f"head_{axis}_approx_deg"] = float(arrays["head_pose"][i, j])
            yield row
    # Stream frames: avoid constructing hundreds of thousands of Python dicts.
    iterator = rows()
    first = next(iterator)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(first))
        writer.writeheader()
        for row in __import__("itertools").chain([first], iterator):
            writer.writerow({k: "" if isinstance(v, float) and not np.isfinite(v) else v for k, v in row.items()})


def run_records(manifest: list[dict], output: Path, model: Path, sessions: list[str] | None = None,
              calibration_dir: Path | None = None, window_s=60.0, stride_s=10.0,
              max_seconds=None, pvt_lapse_ms=500.0):
    """Create reproducible extraction caches, analysis tables and local audit PNGs."""
    from .offline_qa import export_qa
    from .dataset_manifest import load_pvt

    output, model = output.resolve(), model.resolve()
    if not model.is_file():
        raise ValueError(f"Missing MediaPipe model: {model}")
    for name, value in (("window_s", window_s), ("stride_s", stride_s), ("pvt_lapse_ms", pvt_lapse_ms)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if max_seconds is not None and (not np.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be finite and positive")
    output.mkdir(parents=True, exist_ok=True)
    config = PERCLOSConfig()
    selected = sessions if sessions is not None else [r["session_id"] for r in manifest if r["video_available"]]
    by_id = {r["session_id"]: r for r in manifest}
    if len(by_id) != len(manifest):
        raise ValueError("Session IDs must be unique within a processing run")
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Select at least one session, without duplicates")
    for sid in selected:
        if sid not in by_id or not by_id[sid]["video_available"]:
            raise ValueError(f"Missing required video for session {sid}")
    all_windows, all_events, calibrations, pvt_rows, provenance, failures = [], [], [], [], {}, []
    for sid in selected:
        session = by_id[sid]
        try:
            calibration, metadata = load_calibration(calibration_dir, session["subject_id"], session["dataset_id"])
            arrays, identity = extract_recording(session, model, output, max_seconds)
            if calibration:
                for reference in metadata["details"]["sources"]:
                    source_identity = reference.get("extraction_identity", {})
                    for key in ("model_sha256", "extractor_sha256", "mediapipe_version", "pipeline_version"):
                        if source_identity.get(key) != identity[key]:
                            raise ValueError(f"Calibration extraction mismatch ({key}); review references with the current extractor")
            timeline = make_timeline(arrays["landmarks"], arrays["timestamps_ms"], arrays["face_found"], calibration, config)
            pvt = load_pvt(session)
            windows = window_metrics(timeline, session, pvt, metadata, window_s, stride_s,
                                     config.min_valid_fraction, pvt_lapse_ms)
            for row in windows:
                mask = ((timeline.timestamps_s >= row["window_start_s"])
                        & (timeline.timestamps_s < row["window_end_s"]))
                for i, axis in enumerate(("pitch", "yaw", "roll")):
                    values = arrays["head_pose"][mask & timeline.face_found, i]
                    values = values[np.isfinite(values)]
                    row[f"head_{axis}_approx_mean_deg"] = float(np.mean(values)) if len(values) else np.nan
                    row[f"head_{axis}_approx_std_deg"] = float(np.std(values)) if len(values) else np.nan
            export_frames(output / "frames" / f"{sid}.csv", arrays, timeline, session, metadata)
            export_qa(Path(session["video_path"]), output / "qa" / sid, arrays, timeline)
            all_windows.extend(windows)
            all_events.extend({**label_fields(session), **event} for event in timeline.events)
            pvt_rows.extend({**label_fields(session), **event} for event in (pvt or []))
            for i, name in enumerate(EYE_NAMES):
                calibrations.append({"dataset_id": session["dataset_id"], "subject_id": session["subject_id"], "session_id": sid, "eye": name,
                                     "status": metadata["status"], "method": metadata["method"],
                                     "ear_open": calibration.ear_open[i] if calibration else None,
                                     "ear_closed": calibration.ear_closed[i] if calibration else None,
                                     "profile_sha256": metadata.get("profile_sha256", "")})
            session.update({"processing_status": "COMPLETE" if max_seconds is None else "PARTIAL_DEBUG",
                            "frame_count": len(timeline.ear), "first_timestamp_s": float(timeline.timestamps_s[0]),
                            "last_timestamp_s": float(timeline.timestamps_s[-1]),
                            "nominal_fps": 1 / timeline.nominal_period_s,
                            "face_found_fraction_frames": float(np.mean(timeline.face_found)),
                            "window_count": len(windows), "calibration_status": metadata["status"]})
            provenance[sid] = {"extraction": identity, "calibration": metadata}
            print(f"{sid}: {len(windows)} windows, face coverage {np.mean(timeline.face_found):.1%}, {metadata['status']}", flush=True)
        except (ValueError, OSError, RuntimeError) as error:
            session.update({"processing_status": "FAILED", "error": str(error)})
            failures.append(f"{sid}: {error}")
            print(f"FAILED {sid}: {error}", flush=True)
        # Checkpoint aggregate outputs after each session. Selected sessions
        # only are included in feature tables; manifest covers missing tests too.
        write_csv(output / "sessions.csv", manifest)
        write_csv(output / "windows.csv", all_windows, fields=WINDOW_FIELDS)
        write_csv(output / "events.csv", all_events, EVENT_FIELDS)
        write_csv(output / "calibration.csv", calibrations, ["dataset_id", "subject_id", "session_id", "eye", "status", "method", "ear_open", "ear_closed", "profile_sha256"])
        write_csv(output / "pvt_events.csv", pvt_rows, [*LABEL_FIELDS, "stimulus_s", "response_s", "reaction_time_ms"])
        (output / "run.json").write_text(json.dumps({
            "pipeline_version": PIPELINE_VERSION, "schema_version": 1, "selected_sessions": selected,
            "window_s": window_s, "stride_s": stride_s, "max_seconds": max_seconds,
            "perclos_config": asdict(config), "pvt_lapse_threshold_ms": pvt_lapse_ms,
            "combined_perclos_method": "arithmetic mean of both available per-eye time fractions",
            "event_count_method": "continuous recording events; observed termination in [start,end); censored excluded",
            "event_duration_method": "existing engine: last closed timestamp minus first plus nominal frame period",
            "sessions": provenance, "inputs": manifest, "failures": failures,
        }, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError("Some sessions failed; inspect sessions.csv: " + "; ".join(failures))
    return manifest, all_windows
