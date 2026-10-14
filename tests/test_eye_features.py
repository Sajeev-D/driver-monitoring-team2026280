import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eye_features.buffer import LandmarkBuffer  # noqa: E402
from eye_features.extractor import FrameResult, _empty_result  # noqa: E402
from eye_features.landmarks import eye_aspect_ratio  # noqa: E402


def make_eye(opening):
    """A fake eye 30 px wide; opening = lid gap in px."""
    half = opening / 2
    return np.array([
        [0, 0],          # p1 corner
        [10, -half],     # p2 upper
        [20, -half],     # p3 upper
        [30, 0],         # p4 corner
        [20, half],      # p5 lower
        [10, half],      # p6 lower
    ], dtype=np.float32)


def make_frame(ts, opening=9.0):
    eye = make_eye(opening)
    return FrameResult(np.stack([eye, eye]), np.zeros(3, np.float32), ts, True)


def test_ear_open_vs_closed():
    assert abs(eye_aspect_ratio(make_eye(9)) - 0.3) < 1e-6
    assert eye_aspect_ratio(make_eye(1)) < 0.05


def test_ear_nan_when_missing():
    assert np.isnan(eye_aspect_ratio(np.full((6, 2), np.nan)))


def test_buffer_is_time_based():
    buf = LandmarkBuffer(window_s=60)
    for i in range(2000):                 # 2000 frames at 30 fps = 66.6 s
        buf.push(make_frame(int(i * 1000 / 30)))
    assert buf.span_s() <= 60.0
    assert 1795 <= len(buf) <= 1801


def test_buffer_handles_dropped_frames():
    buf = LandmarkBuffer(window_s=60)
    for i in range(0, 2000, 2):           # only every second frame arrives
        buf.push(make_frame(int(i * 1000 / 30)))
    assert buf.span_s() <= 60.0
    assert 895 <= len(buf) <= 901         # about half as many frames, same 60 s


def test_face_lost_frames_are_kept_as_nan():
    buf = LandmarkBuffer()
    buf.push(make_frame(0))
    buf.push(_empty_result(33))
    buf.push(make_frame(66))
    arrays = buf.as_arrays()
    assert arrays["landmarks"].shape == (3, 2, 6, 2)
    assert arrays["face_found"].tolist() == [True, False, True]
    assert np.isnan(arrays["landmarks"][1]).all()
    assert abs(buf.missing_fraction() - 1 / 3) < 1e-9
    ears = buf.ear_series()
    assert ears.shape == (3, 2) and np.isnan(ears[1]).all()


def test_empty_buffer_shapes():
    arrays = LandmarkBuffer().as_arrays()
    assert arrays["landmarks"].shape == (0, 2, 6, 2)
