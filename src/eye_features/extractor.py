"""Per-frame eye feature extraction with MediaPipe Face Landmarker.

Usage:
    with EyeFeatureExtractor("models/face_landmarker.task") as extractor:
        result = extractor.process(frame_bgr, timestamp_ms)
"""

import math
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

from .landmarks import EYES


@dataclass
class FrameResult:
    """Output for one frame.

    landmarks:  float32 [2, 6, 2], pixel (x, y). [0] = right eye, [1] = left eye.
                All NaN if no face was found.
    head_pose:  float32 [3], (pitch, yaw, roll) in degrees. NaN if no face.
    timestamp_ms: int, the timestamp passed in for this frame.
    face_found: bool
    """

    landmarks: np.ndarray
    head_pose: np.ndarray
    timestamp_ms: int
    face_found: bool


def _empty_result(timestamp_ms):
    return FrameResult(
        landmarks=np.full((2, 6, 2), np.nan, dtype=np.float32),
        head_pose=np.full(3, np.nan, dtype=np.float32),
        timestamp_ms=int(timestamp_ms),
        face_found=False,
    )


def head_pose_from_matrix(matrix):
    """Approximate (pitch, yaw, roll) in degrees from MediaPipe's 4x4 facial
    transformation matrix. Good enough for "is the head turned or dropping";
    check the sign conventions on real data before relying on them."""
    r = np.asarray(matrix, dtype=np.float64)[:3, :3]
    # Remove any scale so r is a pure rotation.
    r = r / np.linalg.norm(r, axis=0, keepdims=True)
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
        roll = math.atan2(r[1, 0], r[0, 0])
    else:
        pitch = math.atan2(-r[1, 2], r[1, 1])
        yaw = math.atan2(-r[2, 0], sy)
        roll = 0.0
    return np.degrees([pitch, yaw, roll]).astype(np.float32)


class EyeFeatureExtractor:
    """Wraps MediaPipe Face Landmarker and returns only what we need."""

    def __init__(self, model_path, min_face_confidence=0.5):
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_face_confidence,
            min_face_presence_confidence=min_face_confidence,
            min_tracking_confidence=min_face_confidence,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, frame, timestamp_ms):
        """frame: BGR (from OpenCV) or single-channel grayscale/NIR image.
        timestamp_ms: must increase every call (MediaPipe VIDEO mode rule)."""
        # MediaPipe needs strictly increasing timestamps.
        timestamp_ms = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = timestamp_ms

        if frame.ndim == 2:  # NIR / grayscale camera
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        return self._to_frame_result(result, rgb.shape[1], rgb.shape[0], timestamp_ms)

    @staticmethod
    def _to_frame_result(result, width, height, timestamp_ms):
        if not result.face_landmarks:
            return _empty_result(timestamp_ms)

        face = result.face_landmarks[0]
        landmarks = np.empty((2, 6, 2), dtype=np.float32)
        for e, indices in enumerate(EYES):
            for k, idx in enumerate(indices):
                # MediaPipe gives x, y as fractions of width/height. Convert to
                # pixels so distances are the same scale in both directions.
                landmarks[e, k, 0] = face[idx].x * width
                landmarks[e, k, 1] = face[idx].y * height

        head_pose = np.full(3, np.nan, dtype=np.float32)
        matrices = getattr(result, "facial_transformation_matrixes", None)
        if matrices:
            head_pose = head_pose_from_matrix(matrices[0])

        return FrameResult(landmarks, head_pose, int(timestamp_ms), True)

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
