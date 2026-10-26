"""Numerical PERCLOS regression tests; do not require a connected camera."""
from pathlib import Path
import importlib.util
import sys

import numpy as np
import pytest

# Import just the numerical module, so these tests can run in a headless CI
# machine without the MediaPipe runtime.
MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "eye_features" / "perclos.py"
spec = importlib.util.spec_from_file_location("perclos_numeric_test", MODULE_PATH)
perclos = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = perclos
spec.loader.exec_module(perclos)


def eye(ear):
    """Synthetic eye with horizontal width 30; vertical opening = ear*30."""
    gap = ear * 30
    half = gap / 2
    return np.array([[0, 0], [10, -half], [20, -half], [30, 0],
                     [20, half], [10, half]], dtype=np.float32)


def make_data(seconds=10, fps=30):
    n = int(seconds * fps)
    data = np.tile(np.stack((eye(.30), eye(.30))), (n, 1, 1, 1))
    ts = np.arange(n, dtype=np.float64) / fps
    return data, ts


def test_all_open_zero_perclos():
    lm, ts = make_data()
    result = perclos.calculate_perclos(lm, ts, (.30, .30), (.05, .05))
    assert result["perclos_raw"] == 0
    assert result["blink_count"] == 0
    assert result["quality_flag"] == "OK"


def test_both_eyes_closed_counts_and_elapsed_time():
    lm, ts = make_data()
    lm[60:66] = np.stack((eye(.05), eye(.05)))   # 0.20s blink
    lm[150:192] = np.stack((eye(.05), eye(.05)))  # 1.40s prolonged closure
    result = perclos.calculate_perclos(lm, ts, (.30, .30), (.05, .05))
    assert result["blink_count"] == 1
    assert abs(result["average_blink_duration_ms"] - 200) < 1e-6
    assert result["prolonged_closure_count"] == 1
    assert 0.15 < result["perclos_raw"] < 0.17
    assert len(result["closure_events"]) == 2


def test_one_eye_closed_does_not_count_as_both_closed():
    lm, ts = make_data()
    lm[80:130, 0] = eye(.05)
    result = perclos.calculate_perclos(lm, ts, (.30, .30), (.05, .05))
    assert result["perclos_raw"] == 0
    assert result["prolonged_closure_count"] == 0


def test_lost_face_not_treated_as_open_or_closed():
    lm, ts = make_data()
    face_lost = np.zeros(len(ts), dtype=bool)
    face_lost[60:90] = True
    lm[60:90] = np.nan
    result = perclos.calculate_perclos(lm, ts, (.30, .30), (.05, .05), face_lost=face_lost)
    assert result["perclos_raw"] == 0
    assert .85 < result["valid_data_fraction"] < .95
    assert result["quality_flag"] == "OK"


def test_quality_low_for_excessive_missing_data():
    lm, ts = make_data()
    lm[50:200] = np.nan
    result = perclos.calculate_perclos(lm, ts, (.30, .30), (.05, .05))
    assert result["quality_flag"] == "LOW_VALID_DATA"
    assert result["valid_data_fraction"] < .8


def test_ongoing_event_is_not_a_confirmed_blink():
    lm, ts = make_data(seconds=2)
    lm[-5:] = np.stack((eye(.05), eye(.05)))
    result = perclos.calculate_perclos(lm, ts, (.30, .30), (.05, .05))
    assert result["blink_count"] == 0
    assert result["closure_events"][-1]["completed"] is False


def test_single_eye_backwards_compatibility():
    lm, ts = make_data()
    result = perclos.calculate_perclos(lm[:, 0], ts, .30, .05)
    assert result["ear"].shape == (len(ts),)
    assert result["per_eye_ear"].shape == (len(ts), 1)
    assert isinstance(result["ear_open"], float)


def test_bad_reference_rejected():
    lm, ts = make_data()
    with pytest.raises(ValueError, match="ear_open > ear_closed"):
        perclos.calculate_perclos(lm, ts, (.10, .30), (.20, .05))


@pytest.mark.parametrize("duration,blinks,longer", [
    (.10, 0, 0), (.15, 1, 0), (.30, 1, 0),
    (.40, 1, 0), (.41, 0, 1), (.60, 0, 1),
])
def test_duration_category_boundaries(duration, blinks, longer):
    lm, ts = make_data(seconds=2, fps=100)
    lm[50:50 + round(duration * 100)] = np.stack((eye(.05), eye(.05)))
    result = perclos.calculate_perclos(lm, ts, (.30, .30), (.05, .05))
    assert result["blink_count"] == blinks
    assert result["prolonged_closure_count"] == longer
    assert result["perclos_raw"] > 0
