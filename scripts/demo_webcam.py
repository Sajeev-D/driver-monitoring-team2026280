"""Live demo of the eye feature extraction module.

Shows the 12 eye landmarks, live EAR for both eyes, a 10 second EAR graph,
face-lost status, head pose, and how full the 60 s buffer is.

Run from the repo root:
    python scripts/demo_webcam.py                  # default webcam
    python scripts/demo_webcam.py --camera 1       # another camera
    python scripts/demo_webcam.py --video clip.mp4 # a recorded video
    python scripts/demo_webcam.py --picamera       # Raspberry Pi CSI camera (OV9281)

Keys:  q or Esc = quit,  s = save the 60 s buffer to recordings/*.npz
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from eye_features import EyeFeatureExtractor, LandmarkBuffer, eye_aspect_ratio  # noqa: E402

GRAPH_SECONDS = 10
GRAPH_HEIGHT = 120
EAR_MAX = 0.45           # top of the graph
DEMO_CLOSED_LINE = 0.20  # only a visual guide, not our real threshold

GREEN = (0, 220, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
YELLOW = (0, 220, 255)
CYAN = (255, 220, 0)
GREY = (140, 140, 140)


def parse_args():
    p = argparse.ArgumentParser(description="Eye feature extraction demo")
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--video", type=str, help="use a video file instead of the webcam")
    p.add_argument("--picamera", action="store_true",
                   help="use the Raspberry Pi CSI camera (e.g. OV9281) through Picamera2")
    p.add_argument("--model", type=str,
                   default=str(REPO_ROOT / "models" / "face_landmarker.task"))
    p.add_argument("--width", type=int, help="default 640")
    p.add_argument("--height", type=int, help="default 480 (400 with --picamera)")
    p.add_argument("--fps", type=float, default=30.0, help="--picamera frame rate (default 30)")
    p.add_argument("--no-mirror", action="store_true",
                   help="don't flip the preview like a mirror")
    args = p.parse_args()
    args.width = args.width or 640
    # OV9281 is 1280x800 (16:10), so 640x400 keeps the full field of view.
    args.height = args.height or (400 if args.picamera else 480)
    return args


class PiCamera:
    """Raspberry Pi CSI camera through Picamera2 (libcamera), with the same
    read() / get() / release() calls the demo uses on cv2.VideoCapture.

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

    def release(self):
        self._cam.stop()
        self._cam.close()


def put_text(img, text, org, color=WHITE, scale=0.6):
    """Text on a dark box so it stays readable over any background."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 3, y + base + 1), (0, 0, 0), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_ear_graph(canvas, buffer):
    """Draw the last GRAPH_SECONDS of EAR for both eyes along the bottom."""
    h, w = canvas.shape[:2]
    top = h - GRAPH_HEIGHT
    cv2.rectangle(canvas, (0, top), (w, h), (30, 30, 30), -1)

    def y_of(ear):
        ear = min(max(ear, 0.0), EAR_MAX)
        return int(h - 5 - (ear / EAR_MAX) * (GRAPH_HEIGHT - 10))

    y_line = y_of(DEMO_CLOSED_LINE)
    cv2.line(canvas, (0, y_line), (w, y_line), GREY, 1)
    put_text(canvas, f"{DEMO_CLOSED_LINE:.2f}", (5, y_line - 4), GREY, 0.45)

    arrays = buffer.as_arrays()
    ts = arrays["timestamps_ms"]
    if len(ts) < 2:
        return
    ears = buffer.ear_series()
    t_end = ts[-1]
    keep = ts >= t_end - GRAPH_SECONDS * 1000
    ts, ears = ts[keep], ears[keep]
    xs = ((ts - (t_end - GRAPH_SECONDS * 1000)) / (GRAPH_SECONDS * 1000) * (w - 1)).astype(int)

    for eye, color in ((0, YELLOW), (1, CYAN)):
        run = []
        for x, ear in zip(xs, ears[:, eye]):
            if np.isnan(ear):          # face lost: break the line
                if len(run) > 1:
                    cv2.polylines(canvas, [np.array(run)], False, color, 2)
                run = []
            else:
                run.append((x, y_of(float(ear))))
        if len(run) > 1:
            cv2.polylines(canvas, [np.array(run)], False, color, 2)

    put_text(canvas, "EAR right eye", (w - 150, top + 18), YELLOW, 0.45)
    put_text(canvas, "EAR left eye", (w - 150, top + 36), CYAN, 0.45)


def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found at {model_path}\n"
              "Download face_landmarker.task (see README) and put it in models/.")
        sys.exit(1)

    if args.video:
        cap = cv2.VideoCapture(args.video)
    elif args.picamera:
        cap = PiCamera(args.width, args.height, args.fps)
    else:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print("Could not open the camera/video. Try --camera 1, or check camera permissions.")
        sys.exit(1)

    buffer = LandmarkBuffer(window_s=60)
    save_dir = REPO_ROOT / "recordings"
    fps = 0.0
    last_time = time.monotonic()
    start = time.monotonic()
    frame_index = 0
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    with EyeFeatureExtractor(model_path) as extractor:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if args.video:
                timestamp_ms = int(frame_index * 1000 / video_fps)
            else:
                timestamp_ms = int((time.monotonic() - start) * 1000)
            frame_index += 1

            result = extractor.process(frame, timestamp_ms)
            buffer.push(result)

            # Draw the eye points on the unflipped frame, then mirror it.
            view = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            view = view.copy()
            if result.face_found:
                for eye in result.landmarks:
                    for x, y in eye:
                        cv2.circle(view, (int(x), int(y)), 2, GREEN, -1)
            if not args.no_mirror:
                view = cv2.flip(view, 1)

            now = time.monotonic()
            dt = now - last_time
            last_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            ear_r = eye_aspect_ratio(result.landmarks[0])
            ear_l = eye_aspect_ratio(result.landmarks[1])

            if result.face_found:
                put_text(view, "FACE: TRACKED", (10, 25), GREEN)
                put_text(view, f"EAR  R {ear_r:.3f}   L {ear_l:.3f}", (10, 50))
                pitch, yaw, roll = result.head_pose
                put_text(view, f"Head  pitch {pitch:+.0f}  yaw {yaw:+.0f}  roll {roll:+.0f}", (10, 75))
            else:
                put_text(view, "FACE: LOST", (10, 25), RED)

            put_text(view, f"FPS {fps:.1f}", (view.shape[1] - 110, 25))
            put_text(view, f"Buffer {len(buffer)} frames / {buffer.span_s():.0f} s   "
                           f"missing {buffer.missing_fraction() * 100:.0f}%", (10, 100))

            draw_ear_graph(view, buffer)
            cv2.imshow("Eye feature extraction demo (q = quit, s = save)", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                save_dir.mkdir(parents=True, exist_ok=True)
                out = save_dir / f"buffer_{time.strftime('%Y%m%d_%H%M%S')}.npz"
                np.savez(out, **buffer.as_arrays())
                print(f"Saved {len(buffer)} frames to {out}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
