"""Raspberry Pi CSI camera (e.g. OV9281) input for the demos."""

import cv2
import numpy as np


class PiCamera:
    """Raspberry Pi CSI camera through Picamera2 (libcamera), with the same
    read() / get() / set() / release() calls the demos use on cv2.VideoCapture.

    Returns single-channel grayscale frames: the Y plane of YUV420, which for
    a mono sensor like the OV9281 is the whole image."""

    def __init__(self, width, height, fps):
        from picamera2 import Picamera2  # only installed on the Pi

        self._cam = Picamera2()
        config = self._cam.create_video_configuration(
            main={"size": (width, height), "format": "YUV420"},
            controls={"FrameRate": fps},
        )
        self._cam.configure(config)
        # libcamera may adjust the size slightly; use what it actually gave us.
        self._width, self._height = self._cam.camera_configuration()["main"]["size"]
        self._fps = fps
        self._cam.start()

    def isOpened(self):
        return True

    def read(self):
        yuv = self._cam.capture_array("main")
        return True, np.ascontiguousarray(yuv[:self._height, :self._width])

    def get(self, prop):
        return self._fps if prop == cv2.CAP_PROP_FPS else 0.0

    def set(self, prop, value):
        return False  # size and frame rate are fixed in __init__

    def release(self):
        self._cam.stop()
        self._cam.close()
