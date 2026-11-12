"""Rolling-buffer and calibration-contract integration tests."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eye_features import EyeCalibration, LandmarkBuffer, compute_rolling_metrics
from eye_features.extractor import FrameResult, _empty_result


def eye(ear):
    d = 15 * ear
    return np.asarray([[0, 0], [10, -d], [20, -d], [30, 0],
                       [20, d], [10, d]], dtype=np.float32)


def frame(ts, r=.30, l=.30):
    return FrameResult(np.stack([eye(r), eye(l)]), np.zeros(3, dtype=np.float32), ts, True)


def test_calibration_json_roundtrip(tmp_path):
    c = EyeCalibration((.30, .28), (.08, .07))
    p = tmp_path / "profile.json"
    c.save(p)
    assert EyeCalibration.load(p) == c
    assert np.allclose(c.p80_ear_thresholds(), (.124, .112))


def test_images_must_have_sufficient_ear_difference():
    with pytest.raises(ValueError, match="too similar"):
        EyeCalibration((.30, .30), (.29, .29))


def test_rolling_buffer_ready_and_drops_old_frames():
    buf = LandmarkBuffer(window_s=60)
    c = EyeCalibration((.30, .30), (.05, .05))
    for i in range(40):
        buf.push(frame(i * 1000))
    partial = compute_rolling_metrics(buf, c)
    assert not partial["window_ready"]
    assert partial["display_quality"] == "WARMING_UP"

    for i in range(40, 75):
        buf.push(frame(i * 1000))
    full = compute_rolling_metrics(buf, c)
    assert 59.0 <= full["window_span_s"] <= 60
    assert full["window_ready"]
    assert full["quality_flag"] == "OK"
    assert full["perclos_percent"] == 0
    assert 60 <= len(buf) <= 61


def test_low_quality_after_tracking_loss():
    buf = LandmarkBuffer(window_s=60)
    c = EyeCalibration((.30, .30), (.05, .05))
    for i in range(61):
        buf.push(_empty_result(i * 1000) if 10 <= i < 40 else frame(i * 1000))
    result = compute_rolling_metrics(buf, c)
    assert result["window_ready"]
    assert result["display_quality"] == "LOW_VALID_DATA"


def test_perclos_averages_eyes_instead_of_requiring_simultaneous_closure():
    buf = LandmarkBuffer(window_s=60)
    c = EyeCalibration((.30, .30), (.05, .05))
    for i in range(61):
        buf.push(frame(i * 1000, r=.05, l=.30))
    result = compute_rolling_metrics(buf, c)
    assert result["eyes"]["right"]["perclos_percent"] == pytest.approx(100)
    assert result["eyes"]["left"]["perclos_percent"] == pytest.approx(0)
    assert result["perclos_percent"] == pytest.approx(50)
    assert result["perclos_raw"] == pytest.approx(.5)
    assert result["closed_time_s"] == 0  # Event metrics remain both-eye.
