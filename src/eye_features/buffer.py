"""Rolling buffer of the last N seconds of FrameResults.

The buffer is time-based, not frame-count based: it keeps every frame whose
timestamp is within `window_s` of the newest one. Frames where the face was
lost are kept (as NaN rows) so later code can tell "missing" apart from
"eyes open" or "eyes closed".
"""

from collections import deque

import numpy as np

from .landmarks import eye_aspect_ratio


class LandmarkBuffer:
    def __init__(self, window_s=60.0):
        self.window_ms = int(window_s * 1000)
        self._frames = deque()

    def push(self, frame_result):
        self._frames.append(frame_result)
        newest = frame_result.timestamp_ms
        while self._frames and newest - self._frames[0].timestamp_ms > self.window_ms:
            self._frames.popleft()

    def __len__(self):
        return len(self._frames)

    def span_s(self):
        """How many seconds of data the buffer currently covers."""
        if len(self._frames) < 2:
            return 0.0
        return (self._frames[-1].timestamp_ms - self._frames[0].timestamp_ms) / 1000.0

    def missing_fraction(self):
        """Fraction of buffered frames where no face was found."""
        if not self._frames:
            return 0.0
        return sum(not f.face_found for f in self._frames) / len(self._frames)

    def as_arrays(self):
        """Return the buffer as numpy arrays (the module's Output B)."""
        n = len(self._frames)
        if n == 0:
            return {
                "landmarks": np.empty((0, 2, 6, 2), dtype=np.float32),
                "head_pose": np.empty((0, 3), dtype=np.float32),
                "timestamps_ms": np.empty(0, dtype=np.int64),
                "face_found": np.empty(0, dtype=bool),
            }
        return {
            "landmarks": np.stack([f.landmarks for f in self._frames]),
            "head_pose": np.stack([f.head_pose for f in self._frames]),
            "timestamps_ms": np.array([f.timestamp_ms for f in self._frames], dtype=np.int64),
            "face_found": np.array([f.face_found for f in self._frames], dtype=bool),
        }

    def ear_series(self):
        """EAR per frame for both eyes, shape [N, 2]. NaN where the face was lost."""
        return np.array(
            [[eye_aspect_ratio(f.landmarks[0]), eye_aspect_ratio(f.landmarks[1])]
             for f in self._frames],
            dtype=np.float32,
        ).reshape(-1, 2)
