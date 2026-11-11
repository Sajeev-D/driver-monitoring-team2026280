"""Cross-dataset input and output contracts, independent of webcam hardware."""

import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eye_features.dataset_manifest import read_manifest, normalize_record, load_recording_timestamps, load_pvt
from eye_features.offline import WINDOW_FIELDS, make_timeline, window_metrics


def record(tmp_path, **extras):
    return normalize_record({"dataset_id": "OUR_DATA", "subject_id": "driver_A",
                             "session_id": "trip_01", "video_path": "trip.mp4", **extras}, tmp_path)


def test_manifest_relative_paths_and_untranslated_labels(tmp_path):
    path = tmp_path / "manifest.csv"
    path.write_text("dataset_id,subject_id,session_id,video_path,drowsiness_label,label_source\n"
                    "CUSTOM,driver_A,trip_01,videos/trip.mp4,awake,manual_review\n")
    rec = read_manifest(path)[0]
    assert rec["video_path"] == str(tmp_path / "videos/trip.mp4")
    assert rec["drowsiness_label"] == "awake"
    assert rec["kss"] is None
    assert rec["label_scope"] == "session"
    assert rec["time_source"] == "video_pts"


def test_manifest_timestamp_units_and_header(tmp_path):
    (tmp_path / "time.csv").write_text("frame,time\n0,0.012\n1,0.078\n2,0.145\n")
    rec = record(tmp_path, timestamps_path="time.csv", timestamp_unit="s",
                 timestamp_column=1, timestamps_skiprows=1)
    assert np.allclose(load_recording_timestamps(rec), [12, 78, 145])
    assert rec["time_source"] == "supplied_timestamps"
    assert record(tmp_path, fps=15)["time_source"] == "explicit_constant_fps"


def test_optional_pvt_missing_is_unavailable_not_zero_observations(tmp_path):
    rec = record(tmp_path)
    assert load_pvt(rec) is None
    lm = np.zeros((2, 2, 6, 2))
    lm[:, :, :, 0] = np.arange(6)
    t = make_timeline(lm, [0, 60000], [False, False])
    row = window_metrics(t, rec, None, {"status": "NO_CALIBRATION", "method": "none"})[0]
    assert row["pvt_available"] is False
    assert np.isnan(row["pvt_response_count"])
    assert row["drowsiness_label"] is None
    assert row["kss"] is None
    assert set(row) <= set(WINDOW_FIELDS)


def test_generic_pvt_times_share_video_origin(tmp_path):
    (tmp_path / "pvt.csv").write_text("stimulus_s,response_s\n1.2,1.55\n")
    rows = load_pvt(record(tmp_path, pvt_path="pvt.csv"))
    assert rows[0]["reaction_time_ms"] == pytest.approx(350)


@pytest.mark.parametrize("extras", [
    {"subject_id": "../driver"}, {"fps": -1}, {"timestamp_unit": "frames"},
    {"label_scope": "frame"}, {"timestamp_column": -1}, {"kss": 10},
    {"timestamps_path": "missing.csv"}, {"extra_labels_json": "[]"},
])
def test_bad_manifest_inputs_are_rejected(tmp_path, extras):
    with pytest.raises(ValueError):
        record(tmp_path, **extras)


def test_duplicate_session_ids_rejected(tmp_path):
    path = tmp_path / "manifest.csv"
    path.write_text("dataset_id,subject_id,session_id,video_path\nA,p1,clip,a.mp4\nA,p2,clip,b.mp4\n")
    with pytest.raises(ValueError, match="unique"):
        read_manifest(path)


@pytest.mark.parametrize("mode", ["fps", "pts"])
def test_video_timing_without_external_timestamp_file(tmp_path, monkeypatch, mode):
    import cv2
    import eye_features.offline as offline
    import eye_features.extractor as extractor
    rec = record(tmp_path, fps=20 if mode == "fps" else None)
    monkeypatch.setattr(offline, "fingerprint", lambda *args: {"model": "test"})

    class Capture:
        def __init__(self, path): self.index = 0
        def isOpened(self): return True
        def read(self):
            self.index += 1
            return (True, np.zeros((10, 10, 3), np.uint8)) if self.index <= 3 else (False, None)
        def get(self, prop): return [12., 75., 141.][self.index - 1]
        def release(self): pass

    class Extractor:
        def __init__(self, model): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def process(self, frame, timestamp_ms):
            return extractor.FrameResult(np.full((2, 6, 2), np.nan), np.full(3, np.nan), timestamp_ms, False)

    monkeypatch.setattr(cv2, "VideoCapture", Capture)
    monkeypatch.setattr(extractor, "EyeFeatureExtractor", Extractor)
    arrays, _ = offline.extract_recording(rec, tmp_path / "model", tmp_path / "out")
    assert arrays["timestamps_ms"].tolist() == ([0, 50, 100] if mode == "fps" else [12, 75, 141])


def test_shared_export_headers_for_different_labels_and_no_full_windows(tmp_path, monkeypatch):
    import eye_features.offline as offline
    import eye_features.offline_qa as qa
    video = tmp_path / "video.mp4"
    video.touch()
    model = tmp_path / "model.task"
    model.touch()
    lm = np.zeros((3, 2, 6, 2)); lm[:, :, :, 0] = np.arange(6)
    arrays = {"landmarks": lm, "timestamps_ms": np.array([0., 30000., 60000.]),
              "face_found": np.zeros(3, dtype=bool), "head_pose": np.full((3, 3), np.nan)}
    monkeypatch.setattr(offline, "extract_recording", lambda *args: (arrays, {"test": True}))
    monkeypatch.setattr(qa, "export_qa", lambda *args: None)
    outputs = []
    for dataset, window in [("A", 60), ("B", 60), ("C", 120)]:
        rec = record(tmp_path, dataset_id=dataset, video_path=str(video), drowsiness_label="awake" if dataset == "B" else None)
        out = tmp_path / dataset
        offline.run_records([rec], out, model, window_s=window)
        with (out / "windows.csv").open() as handle:
            reader = csv.DictReader(handle)
            outputs.append(reader.fieldnames)
            rows = list(reader)
        assert len(rows) == (0 if dataset == "C" else 1)
    assert outputs[0] == outputs[1] == outputs[2] == WINDOW_FIELDS
