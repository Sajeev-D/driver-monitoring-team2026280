"""Offline integration tests: timestamps, window integration, labels and events."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eye_features.calibration import EyeCalibration
from eye_features.drozy import discover_sessions, read_pvt, read_timestamps
from eye_features.offline import (export_frames, load_calibration, make_timeline, window_metrics, write_csv, calibration_from_reviewed_frames)


SESSION = {"subject_id": 1, "session_id": "1-1", "test_id": 1,
           "kss": 3, "kss_label_scope": "session"}
META = {"status": "VERIFIED_REFERENCES", "method": "reviewed_video_frames"}
CALIBRATION = EyeCalibration((.30, .30), (.05, .05))


def eye(ear):
    h = 15 * ear
    return np.array([[0, 0], [10, -h], [20, -h], [30, 0], [20, h], [10, h]])


def data(seconds=70, fps=10):
    ts = np.arange(round(seconds * fps) + 1) * (1000 / fps)
    landmarks = np.tile(np.stack([eye(.30), eye(.30)]), (len(ts), 1, 1, 1))
    return landmarks, ts, np.ones(len(ts), dtype=bool)


def test_ms_conversion_full_windows_and_known_time_fraction():
    lm, ts, found = data()
    lm[(ts >= 10000) & (ts < 40000)] = np.stack([eye(.05), eye(.05)])
    timeline = make_timeline(lm, ts, found, CALIBRATION)
    assert timeline.timestamps_s[-1] == 70
    rows = window_metrics(timeline, SESSION, [], META)
    assert len(rows) == 2
    assert rows[0]["window_end_s"] == 60
    assert rows[0]["perclos_combined"] == pytest.approx(.5)
    assert rows[0]["valid_data_fraction"] == 1
    assert rows[0]["kss"] == 3
    assert rows[0]["kss_label_scope"] == "session"


def test_time_weighting_and_clipped_interval_at_boundary():
    lm, _, found = data(seconds=3, fps=1)
    ts = np.array([0, 700, 1700, 3000])
    lm[1] = np.stack([eye(.05), eye(.05)])
    timeline = make_timeline(lm, ts, found, CALIBRATION)
    rows = window_metrics(timeline, SESSION, [], META, window_s=1, stride_s=1)
    assert rows[0]["perclos_combined"] == pytest.approx(.3)
    assert rows[1]["perclos_combined"] == pytest.approx(.7)
    assert rows[0]["valid_time_s"] == pytest.approx(1)


def test_gap_and_missing_face_are_not_open_or_closed_time():
    lm, ts, found = data(seconds=60)
    keep = ~((ts > 20000) & (ts < 30000))
    lm, ts, found = lm[keep], ts[keep], found[keep]
    found[(ts >= 40000) & (ts < 50000)] = False
    row = window_metrics(make_timeline(lm, ts, found, CALIBRATION), SESSION, [], META)[0]
    assert row["perclos_combined"] == 0
    assert row["valid_time_s"] == pytest.approx(39.9)
    assert row["tracking_quality"] == "LOW_VALID_DATA"
    assert row["missing_time_s"] == pytest.approx(20.1)


def test_all_missing_does_not_report_zero_metrics():
    lm, ts, found = data(seconds=60)
    found[:] = False
    row = window_metrics(make_timeline(lm, ts, found, CALIBRATION), SESSION, [], META)[0]
    assert np.isnan(row["perclos_combined"])
    assert np.isnan(row["candidate_blink_count_both"])
    assert row["valid_time_s"] == 0


def test_no_calibration_keeps_ear_but_not_closure_metrics():
    lm, ts, found = data(seconds=60)
    timeline = make_timeline(lm, ts, found)
    row = window_metrics(timeline, SESSION, [], {"status": "NO_CALIBRATION", "method": "none"})[0]
    assert row["ear_right_mean"] == pytest.approx(.30)
    assert np.isnan(row["perclos_combined"])
    assert np.isnan(row["candidate_blink_count_both"])
    assert row["quality_status"] == "NO_CALIBRATION"
    assert row["tracking_quality"] == "OK"


def test_per_eye_retained_and_combined_requires_both():
    lm, ts, found = data(seconds=60)
    lm[:, 0] = eye(.05)
    row = window_metrics(make_timeline(lm, ts, found, CALIBRATION), SESSION, [], META)[0]
    assert row["perclos_right"] == 1
    assert row["perclos_left"] == 0
    assert row["perclos_combined"] == .5
    assert row["eye_disagreement_time_s"] == 60
    lm[:, 1] = np.nan
    row = window_metrics(make_timeline(lm, ts, found, CALIBRATION), SESSION, [], META)[0]
    assert row["perclos_right"] == 1
    assert np.isnan(row["perclos_left"])
    assert np.isnan(row["perclos_combined"])


def test_events_are_not_redetected_at_window_boundaries():
    lm, ts, found = data(seconds=3, fps=100)
    lm[(ts >= 900) & (ts < 1100)] = np.stack([eye(.05), eye(.05)])
    timeline = make_timeline(lm, ts, found, CALIBRATION)
    rows = window_metrics(timeline, SESSION, [], META, window_s=1, stride_s=1)
    assert rows[0]["candidate_blink_count_both"] == 0
    assert rows[1]["candidate_blink_count_both"] == 1
    assert rows[1]["mean_blink_duration_both_ms"] == pytest.approx(200)
    assert sum(e["candidate_blink"] for e in timeline.events if e["eye"] == "both") == 1


def test_left_right_and_gap_censored_events_are_not_blinks():
    lm, ts, found = data(seconds=3, fps=100)
    for start, end in [(0, 200), (1000, 1200), (2800, 3010)]:
        lm[(ts >= start) & (ts < end)] = np.stack([eye(.05), eye(.05)])
    found[(ts >= 990) & (ts < 1000)] = False
    timeline = make_timeline(lm, ts, found, CALIBRATION)
    both = [e for e in timeline.events if e["eye"] == "both"]
    assert len(both) == 3
    assert all(not e["candidate_blink"] for e in both)
    assert both[0]["left_censored"] and both[-1]["right_censored"]


def test_startup_prefix_does_not_make_full_window():
    lm, ts, found = data(seconds=18)
    assert window_metrics(make_timeline(lm, ts, found), SESSION, [], META) == []


def test_pvt_header_alignment_and_no_future_response(tmp_path):
    path = tmp_path / "pvt.csv"
    path.write_text("2014-11-26_10.08.39.274\n2014-11-26_10.08.45.613;2014-11-26_10.08.45.926\n")
    pvt = read_pvt(path)
    assert pvt[0]["stimulus_s"] == pytest.approx(6.339)
    assert pvt[0]["reaction_time_ms"] == pytest.approx(313)
    lm, ts, found = data(seconds=60)
    pvt.append({"stimulus_s": 59.9, "response_s": 60.3, "reaction_time_ms": 400})
    row = window_metrics(make_timeline(lm, ts, found), SESSION, pvt, META)[0]
    assert row["pvt_stimulus_count"] == 2
    assert row["pvt_response_count"] == 1
    assert row["pvt_mean_rt_ms"] == pytest.approx(313)


def test_timestamp_validation(tmp_path):
    path = tmp_path / "timestamps.txt"
    path.write_text("2014 11 26 10 8 39 297 23\n2014 11 26 10 8 39 331 57\n")
    assert read_timestamps(path).tolist() == [23, 57]
    path.write_text("2014 11 26 10 8 39 297 23\n2014 11 26 10 8 39 297 23\n")
    with pytest.raises(ValueError, match="increasing"):
        read_timestamps(path)


def test_discovery_preserves_missing_sessions_and_placeholder_kss(tmp_path):
    (tmp_path / "videos_i8").mkdir()
    (tmp_path / "videos_i8" / "1-1.mp4").touch()
    kss = np.full((14, 3), 3)
    kss[6, 0] = 0
    np.savetxt(tmp_path / "KSS.txt", kss, fmt="%d")
    sessions = discover_sessions(tmp_path)
    assert len(sessions) == 42
    assert sessions[0]["video_available"]
    assert sessions[18]["kss"] is None
    assert sessions[18]["processing_status"] == "MISSING_RECORDING"


def test_calibration_requires_review_and_matching_subject(tmp_path):
    path = tmp_path / "subject_1.json"
    path.write_text(json.dumps({"subject_id": 1, "ear_open": [.3, .3], "ear_closed": [.05, .05]}))
    calibration, meta = load_calibration(tmp_path, 1)
    assert calibration is None
    assert meta["status"] == "UNVERIFIED_REFERENCES"
    data = json.loads(path.read_text())
    data.update({"verified_open_and_closed": True, "reviewer": "test", "reviewed_at": "2026-10-25", "sources": [{"session_id": "1-1", "frame_index": 10}]})
    path.write_text(json.dumps(data))
    calibration, meta = load_calibration(tmp_path, 1)
    assert calibration == CALIBRATION
    assert len(meta["profile_sha256"]) == 64
    data["subject_id"] = 2
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="subject mismatch"):
        load_calibration(tmp_path, 1)


def test_csv_nulls_and_missing_p80_status(tmp_path):
    path = tmp_path / "null.csv"
    write_csv(path, [{"a": np.nan, "b": None, "c": 0}])
    assert path.read_text().splitlines()[1] == ",,0"
    lm, ts, found = data(seconds=1)
    found[2] = False
    timeline = make_timeline(lm, ts, found, CALIBRATION)
    arrays = {"landmarks": lm, "timestamps_ms": ts, "face_found": found, "head_pose": np.zeros((len(ts), 3))}
    export_frames(tmp_path / "frames.csv", arrays, timeline, SESSION, META)
    import csv
    rows = list(csv.DictReader((tmp_path / "frames.csv").open()))
    assert rows[2]["p80_closed_right"] == ""
    assert rows[2]["closure_fraction_right"] == ""
    assert rows[0]["p80_closed_right"] == "False"


def test_reviewed_cache_calibration_roundtrip_and_subject_guard(tmp_path):
    lm, ts, found = data(seconds=1)
    lm[-1] = np.stack([eye(.05), eye(.05)])
    np.savez(tmp_path / "1-1.npz", landmarks=lm, timestamps_ms=ts, face_found=found,
             identity=json.dumps({"model_sha256": "test", "subject_id": "1", "dataset_id": "TEST"}))
    profile = calibration_from_reviewed_frames(tmp_path, 1, "1-1", 0, "1-1", len(ts)-1, "Researcher")
    (tmp_path / "subject_1.json").write_text(json.dumps(profile))
    calibration, _ = load_calibration(tmp_path, 1)
    assert calibration.ear_open == pytest.approx((.3, .3))
    assert profile["sources"][1]["timestamp_ms"] == 1000
    with pytest.raises(ValueError, match="does not belong"):
        calibration_from_reviewed_frames(tmp_path, 2, "1-1", 0, "1-1", 1, "Researcher")
    with pytest.raises(ValueError, match="outside"):
        calibration_from_reviewed_frames(tmp_path, 1, "1-1", -1, "1-1", 1, "Researcher")


@pytest.mark.parametrize("window,stride", [(0, 10), (60, 0), (float("nan"), 10)])
def test_invalid_window_configuration(window, stride):
    lm, ts, found = data(seconds=1)
    with pytest.raises(ValueError, match="finite"):
        window_metrics(make_timeline(lm, ts, found), SESSION, [], META, window, stride)


@pytest.mark.parametrize("decoded_count", [2, 3, 4])
def test_extraction_alignment_and_reusable_cache(tmp_path, monkeypatch, decoded_count):
    import cv2
    import eye_features.offline as drozy
    from eye_features.dataset_manifest import normalize_record
    import eye_features.extractor as extractor

    root = tmp_path / "dataset"
    (root / "timestamps").mkdir(parents=True)
    (root / "timestamps/1-1.txt").write_text("\n".join(
        f"2014 11 26 10 8 39 {i*100} {i*100}" for i in range(3)))
    record = normalize_record({"dataset_id": "TEST", "subject_id": "1", "session_id": "1-1", "video_path": "video.mp4", "timestamps_path": "timestamps/1-1.txt", "timestamp_column": 7}, root)
    identity = {"test_model": "stable"}
    monkeypatch.setattr(drozy, "fingerprint", lambda *args: dict(identity))

    class Capture:
        def __init__(self, path):
            self.index = 0
        def isOpened(self):
            return True
        def read(self):
            self.index += 1
            return (True, np.zeros((10, 10, 3), dtype=np.uint8)) if self.index <= decoded_count else (False, None)
        def release(self):
            pass

    class Extractor:
        def __init__(self, model):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def process(self, frame, timestamp_ms):
            return extractor.FrameResult(np.stack([eye(.30), eye(.30)]), np.zeros(3), timestamp_ms, timestamp_ms != 100)

    monkeypatch.setattr(cv2, "VideoCapture", Capture)
    monkeypatch.setattr(extractor, "EyeFeatureExtractor", Extractor)
    output = tmp_path / "out"
    if decoded_count != 3:
        with pytest.raises(ValueError, match="timestamp"):
            drozy.extract_recording(record, tmp_path / "model", output)
        assert not (output / "cache/1-1.npz").exists()
    else:
        arrays, _ = drozy.extract_recording(record, tmp_path / "model", output)
        assert arrays["timestamps_ms"].tolist() == [0, 100, 200]
        assert arrays["face_found"].tolist() == [True, False, True]
        def must_not_decode(*args):
            raise AssertionError("A valid cache should skip decoding")
        monkeypatch.setattr(cv2, "VideoCapture", must_not_decode)
        cached, _ = drozy.extract_recording(record, tmp_path / "model", output)
        assert np.array_equal(cached["landmarks"], arrays["landmarks"])
        identity["test_model"] = "changed"
        with pytest.raises(AssertionError, match="skip decoding"):
            drozy.extract_recording(record, tmp_path / "model", output)


def test_minmax_calibration_and_openness(tmp_path):
    from eye_features.offline import calibration_from_minmax
    lm, ts, found = data(seconds=4, fps=1)
    for i, pair in enumerate(((.2, .25), (.1, .15), (.3, .35), (.9, .9), (.4, .4))):
        lm[i] = np.stack([eye(pair[0]), eye(pair[1])])
    found[3] = False  # Invalid tracking must not become the maximum.
    np.savez(tmp_path / "1-1.npz", landmarks=lm, timestamps_ms=ts, face_found=found,
             identity=json.dumps({"subject_id": "1", "dataset_id": "DROZY"}))
    profile = calibration_from_minmax(tmp_path, "1", "1-1", 0, 2)
    (tmp_path / "subject_1.json").write_text(json.dumps(profile))
    calibration, meta = load_calibration(tmp_path, "1", "DROZY")
    assert meta["status"] == "MINMAX_REFERENCES"
    assert calibration.ear_closed == pytest.approx((.1, .15))
    assert calibration.ear_open == pytest.approx((.3, .35))
    np.testing.assert_allclose(calibration.openness([[.1, .15], [.2, .25], [.3, .35]]),
                               [[0, 0], [.5, .5], [1, 1]], atol=1e-14)
    profile = calibration_from_minmax(tmp_path, "1", "1-1", 0, 4)
    assert profile["ear_open"] == pytest.approx((.4, .4))
    with pytest.raises(ValueError, match="subject mismatch"):
        calibration_from_minmax(tmp_path, "2", "1-1", 0, 2)
    with pytest.raises(ValueError, match="outside"):
        calibration_from_minmax(tmp_path, "1", "1-1", 0, 10)


def test_window_openness_uses_clipped_samples():
    lm, ts, found = data()
    lm[(ts >= 10000) & (ts < 40000)] = np.stack([eye(.05), eye(.05)])
    timeline = make_timeline(lm, ts, found, CALIBRATION)
    row = window_metrics(timeline, SESSION, [], META)[0]
    assert row["eye_openness"] == pytest.approx(.5)
    assert row["eye_openness_right"] == pytest.approx(.5)
    row = window_metrics(make_timeline(lm, ts, found), SESSION, [],
                         {"status": "NO_CALIBRATION", "method": "none"})[0]
    assert np.isnan(row["eye_openness"])
