"""Image-based, per-eye EAR reference calibration for the webcam demo.

The second reference MUST show fully closed (or nearly closed) eyelids.
A generic sleepy-looking face does not establish a closed-eye reference.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from .extractor import EyeFeatureExtractor
from .landmarks import eye_aspect_ratio


@dataclass(frozen=True)
class EyeCalibration:
    # Index 0 = person's right eye, 1 = person's left eye.
    ear_open: tuple[float, float]
    ear_closed: tuple[float, float]

    def __post_init__(self):
        op = np.asarray(self.ear_open, dtype=float)
        cl = np.asarray(self.ear_closed, dtype=float)
        if op.shape != (2,) or cl.shape != (2,) or not np.all(np.isfinite(op)) or not np.all(np.isfinite(cl)):
            raise ValueError("Calibration must contain two finite EAR values per state.")
        if np.any(op <= cl):
            raise ValueError("Open-eye EAR must exceed closed-eye EAR for both eyes.")
        if np.any(op - cl < 0.02):
            raise ValueError("Open-eye and closed-eye images are too similar (EAR gap < 0.02). "
                             "Use an image with clearly open eyes and one with closed eyes.")

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "version": 1,
            "description": "Per-eye open/closed EAR references for P80 (closed-image eyelids must be closed)",
            "ear_open": list(self.ear_open),
            "ear_closed": list(self.ear_closed),
        }, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("version") != 1:
            raise ValueError("Unsupported calibration profile version.")
        return cls(tuple(map(float, data["ear_open"])),
                   tuple(map(float, data["ear_closed"])))

    def p80_ear_thresholds(self):
        op, cl = np.asarray(self.ear_open), np.asarray(self.ear_closed)
        return (op - 0.8 * (op - cl)).tolist()


def calibrate_from_images(model_path, open_image, closed_image):
    """Detect EAR with MediaPipe IMAGE mode (not VIDEO mode) for two reference photos.

    The open image must show clearly open eyes, and the closed image must show
    fully closed (or nearly closed) eyelids. A half-open photo yields an invalid
    P80 baseline.
    """
    images = []
    for label, image_path in (("open", open_image), ("closed", closed_image)):
        path = Path(image_path)
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Could not read {label} image: {path}")
        images.append((label, frame))

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    ears = []
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    try:
        for label, frame in images:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(image)
            frame_result = EyeFeatureExtractor._to_frame_result(
                result, frame.shape[1], frame.shape[0], 0
            )
            if not frame_result.face_found:
                raise ValueError(f"Face not detected in the {label} reference image. "
                                 "Use a clearly visible, forward-facing face.")
            values = tuple(float(eye_aspect_ratio(eye)) for eye in frame_result.landmarks)
            if not all(np.isfinite(v) for v in values):
                raise ValueError(f"Could not calculate both eye EARs in the {label} reference image.")
            ears.append(values)
    finally:
        landmarker.close()
    calibration = EyeCalibration(ear_open=ears[0], ear_closed=ears[1])
    return calibration
