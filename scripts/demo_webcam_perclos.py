"""Live, privacy-conscious eye-monitoring PERCLOS demo.

Run from repository root on macOS:
 python scripts/demo_webcam_perclos.py --open_image calibration/open.jpg --closed_image calibration/closed.jpg
Next time:
 python scripts/demo_webcam_perclos.py

The closed reference MUST show fully or nearly closed eyelids for P80.
Keys: q/Esc quit; s saves the current 60s landmark buffer (not video).
"""

import argparse
import csv
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from eye_features import (  # noqa: E402
    EyeCalibration, EyeFeatureExtractor, LandmarkBuffer, calibrate_from_images,
    compute_rolling_metrics, eye_aspect_ratio,
)

WINDOW_S = 60
GRAPH_S = 10
GRAPH_H = 135
PANEL_W = 360
CALIBRATION_FILE = REPO_ROOT / "calibration" / "eye_calibration.json"

GREEN = (85, 125, 65)
RED = (65, 65, 175)
WHITE = (45, 45, 45)
YELLOW = (60, 125, 165)
CYAN = (155, 105, 65)
GREY = (115, 115, 115)
BG = (245, 245, 245)

CSV_FIELDS = [
    "elapsed_s", "window_span_s", "window_ready", "quality_flag",
    "perclos_raw", "perclos_percent", "blink_count_60s",
    "average_blink_duration_ms", "prolonged_closure_count_60s",
    "valid_data_fraction", "face_missing_fraction", "valid_time_s", "closed_time_s",
    "face_found_now", "ear_right_now", "ear_left_now",
    "head_pitch_deg", "head_yaw_deg", "head_roll_deg",
    "fps_estimate", "ear_open_right", "ear_open_left",
    "ear_closed_right", "ear_closed_left", "p80_ear_right", "p80_ear_left",
]


def parse_args():
    p = argparse.ArgumentParser(description="Live eye tracker + calibrated rolling 60s PERCLOS")
    p.add_argument("--camera", type=int, default=0, help="Webcam index; default 0")
    p.add_argument("--video", type=Path, help="Optional recorded video instead of webcam")
    p.add_argument("--model", type=Path, default=REPO_ROOT / "models" / "face_landmarker.task")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--no-mirror", action="store_true", help="Do not mirror preview")
    p.add_argument("--open_image", "--open-image", "--alert-image", dest="open_image",
                   type=Path, help="Image of the driver with eyes fully open")
    p.add_argument("--closed_image", "--closed-image", "--drowsy-image", dest="closed_image",
                   type=Path, help="Image of the driver with eyelids fully closed")
    p.add_argument("--calibration", type=Path, default=CALIBRATION_FILE,
                   help="Existing calibration JSON; default calibration/eye_calibration.json")
    p.add_argument("--save-calibration", type=Path, default=CALIBRATION_FILE,
                   help="Where to save EAR references when two images are supplied")
    p.add_argument("--log", type=Path, help="CSV path; defaults to recordings/metrics_<timestamp>.csv")
    p.add_argument("--no-log", action="store_true", help="Disable automatic CSV export")
    return p.parse_args()


def load_calibration(args):
    if bool(args.open_image) != bool(args.closed_image):
        raise ValueError("Supply BOTH --open_image and --closed_image for new calibration.")
    if args.open_image:
        calibration = calibrate_from_images(args.model, args.open_image, args.closed_image)
        calibration.save(args.save_calibration)
        print(f"Calibration saved to {args.save_calibration}")
    elif args.calibration.is_file():
        calibration = EyeCalibration.load(args.calibration)
        print(f"Calibration loaded from {args.calibration}")
    else:
        raise ValueError(
            "No calibration found. First run with:\n"
            "  python scripts/demo_webcam_perclos.py --open_image calibration/open.jpg "
            "--closed_image calibration/closed.jpg\n"
            "NOTE: The closed reference must show fully or nearly closed eyelids."
        )
    print(f"EAR open:   R={calibration.ear_open[0]:.3f} L={calibration.ear_open[1]:.3f}")
    print(f"EAR closed: R={calibration.ear_closed[0]:.3f} L={calibration.ear_closed[1]:.3f}")
    print(f"P80 EAR:    R={calibration.p80_ear_thresholds()[0]:.3f} "
          f"L={calibration.p80_ear_thresholds()[1]:.3f}")
    for open_val, closed_val in zip(calibration.ear_open, calibration.ear_closed):
        if closed_val / open_val > 0.65:
            print("WARNING: Closed photo may not show fully closed eyelids. "
                  "P80 thresholds are unreliable unless both lids were closed.")
            break
    return calibration


def text(img, label, xy, color=WHITE, scale=0.56, thickness=1):
    cv2.putText(img, label, xy, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def fmt(value, digits=1, unit=""):
    return f"{value:.{digits}f}{unit}" if value is not None and math.isfinite(value) else "--"


def draw_graph(view, buffer, calibration):
    h, w = view.shape[:2]
    top = h - GRAPH_H
    cv2.rectangle(view, (0, top), (w - 1, h - 1), BG, -1)
    cv2.line(view, (0, top), (w, top), GREY, 1)
    text(view, "EAR - last 10 s", (12, top + 24), WHITE, 0.52)
    text(view, "R", (w - 90, top + 24), YELLOW, 0.52)
    text(view, "L", (w - 45, top + 24), CYAN, 0.52)

    arrays = buffer.as_arrays()
    ts = arrays["timestamps_ms"]
    if len(ts) < 2:
        return
    ears = buffer.ear_series()
    mask = ts >= ts[-1] - GRAPH_S * 1000
    ts, ears = ts[mask], ears[mask]
    if len(ts) < 2:
        return
    baseline = top + 37
    maximum = 0.5
    graph_h = max(10, GRAPH_H - 50)

    def y_for(val):
        return int(baseline + graph_h * (1 - np.clip(val / maximum, 0, 1)))

    xs = ((ts - (ts[-1] - 1000 * GRAPH_S)) /
          (1000 * GRAPH_S) * (w - 1)).astype(int)
    for eye, color in ((0, YELLOW), (1, CYAN)):
        threshold = calibration.p80_ear_thresholds()[eye]
        cv2.line(view, (0, y_for(threshold)), (w, y_for(threshold)), color, 1)
        line = []
        for x, ear in zip(xs, ears[:, eye]):
            if np.isfinite(ear):
                line.append((int(x), y_for(float(ear))))
            else:
                if len(line) > 1:
                    cv2.polylines(view, [np.asarray(line, dtype=np.int32)], False, color, 2)
                line = []
        if len(line) > 1:
            cv2.polylines(view, [np.asarray(line, dtype=np.int32)], False, color, 2)


def draw_panel(height, metrics, calibration, buffer, face_found, right_ear, left_ear, fps, pose):
    panel = np.full((height, PANEL_W, 3), BG, dtype=np.uint8)
    cv2.line(panel, (0, 0), (0, height - 1), (210, 210, 210), 1)
    text(panel, "Eye tracker", (20, 34), WHITE, 0.65)
    text(panel, "Rolling window: 60 s", (20, 60), GREY, 0.46)

    def row(label, value, y, color=WHITE):
        text(panel, label, (20, y), GREY, 0.46)
        (width, _), _ = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        text(panel, value, (PANEL_W - 20 - width, y), color, 0.48)

    def divider(y):
        cv2.line(panel, (20, y), (PANEL_W - 20, y), (215, 215, 215), 1)

    divider(77)
    text(panel, "Mean PERCLOS", (20, 105), GREY, 0.48)
    value = fmt(metrics["perclos_percent"], 1, "%") if metrics else "--"
    text(panel, value, (20, 150), WHITE, 1.15)
    if metrics:
        status = metrics["display_quality"]
        label = {"OK": "Ready", "WARMING_UP": "Collecting first minute",
                 "LOW_VALID_DATA": "Low tracking coverage"}.get(status, status)
        status_color = GREEN if status == "OK" else RED if status == "LOW_VALID_DATA" else YELLOW
        text(panel, label, (20, 178), status_color, 0.46)
        eyes = metrics["eyes"]
        row("Right / left PERCLOS",
            f"{fmt(eyes['right']['perclos_percent'])}% / {fmt(eyes['left']['perclos_percent'])}%", 211)
    else:
        text(panel, "Waiting for camera samples", (20, 178), GREY, 0.46)
    divider(230)
    text(panel, "Both-eye events", (20, 257), WHITE, 0.49)
    rows = [
        ("Blinks (150-400 ms)", str(metrics["blink_count"]) if metrics else "--"),
        ("Mean blink duration", fmt(metrics["average_blink_duration_ms"], 0, " ms") if metrics else "--"),
        ("Longer closures (>400 ms)", str(metrics["prolonged_closure_count"]) if metrics else "--"),
        ("Valid tracking", fmt(metrics["valid_data_fraction"] * 100, 1, "%") if metrics else "--"),
        ("Face missing", fmt(buffer.missing_fraction() * 100, 1, "%")),
        ("Window covered", fmt(metrics["window_span_s"], 1, " s") if metrics else "--"),
    ]
    for index, (label, value) in enumerate(rows):
        row(label, value, 285 + index * 26)
    divider(431)
    row("Camera", "Tracking" if face_found else "Face not found", 459,
        GREEN if face_found else RED)
    row("Frame rate", fmt(fps, 1, " fps"), 485)
    row("EAR right / left", f"{fmt(right_ear, 3)} / {fmt(left_ear, 3)}", 511)
    row("P80 threshold R / L", " / ".join(f"{v:.3f}" for v in calibration.p80_ear_thresholds()), 537)
    row("Head pitch / yaw / roll", " / ".join(fmt(v, 0) for v in pose), 563)
    text(panel, "q / Esc  Quit     s  Save landmarks", (20, height - 19), GREY, 0.42)
    return panel


def open_capture(args):
    if args.video:
        return cv2.VideoCapture(str(args.video))
    if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
        cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(args.camera)


def make_csv_row(metrics, calibration, frame, ears, fps):
    pose = frame.head_pose
    base = {
        "elapsed_s": round(frame.timestamp_ms / 1000, 3),
        "window_span_s": round(metrics["window_span_s"], 3),
        "window_ready": metrics["window_ready"],
        "quality_flag": metrics["display_quality"],
        "perclos_raw": metrics["perclos_raw"],
        "perclos_percent": metrics["perclos_percent"],
        "blink_count_60s": metrics["blink_count"],
        "average_blink_duration_ms": metrics["average_blink_duration_ms"],
        "prolonged_closure_count_60s": metrics["prolonged_closure_count"],
        "valid_data_fraction": metrics["valid_data_fraction"],
        "face_missing_fraction": metrics["face_missing_fraction"],
        "valid_time_s": metrics["valid_time_s"],
        "closed_time_s": metrics["closed_time_s"],
        "face_found_now": frame.face_found,
        "ear_right_now": ears[0], "ear_left_now": ears[1],
        "head_pitch_deg": pose[0], "head_yaw_deg": pose[1], "head_roll_deg": pose[2],
        "fps_estimate": fps,
        "ear_open_right": calibration.ear_open[0], "ear_open_left": calibration.ear_open[1],
        "ear_closed_right": calibration.ear_closed[0], "ear_closed_left": calibration.ear_closed[1],
        "p80_ear_right": calibration.p80_ear_thresholds()[0],
        "p80_ear_left": calibration.p80_ear_thresholds()[1],
    }
    # Empty cells instead of writing NaN tokens into CSV.
    return {k: ("" if isinstance(v, (float, np.floating)) and not math.isfinite(v)
                else v) for k, v in base.items()}


def main():
    args = parse_args()
    if not args.model.is_file():
        raise SystemExit(f"Model not found: {args.model}")
    try:
        calibration = load_calibration(args)
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f"Calibration error: {exc}") from exc

    cap = open_capture(args)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera/video. On Mac check System Settings > Privacy & Security > "
                         "Camera permission for Terminal/VS Code, or use --camera 1.")
    if not args.video:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(video_fps) or video_fps < 1:
        video_fps = 30.0

    buffer = LandmarkBuffer(window_s=WINDOW_S)
    path = args.log or (REPO_ROOT / "recordings" /
                        f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    csv_file = None
    csv_writer = None
    last_calc_ms = -1000
    last_log_ms = -1000
    metrics = None
    fps = 0.0
    last_wall = time.monotonic()
    start = time.monotonic()
    count = 0

    try:
        if not args.no_log:
            path.parent.mkdir(parents=True, exist_ok=True)
            csv_file = path.open("w", newline="", encoding="utf-8")
            csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            csv_writer.writeheader()
            print(f"Saving per-second metrics (no images/video) to {path}")
        with EyeFeatureExtractor(args.model) as extractor:
            while True:
                ok, raw_frame = cap.read()
                if not ok:
                    break
                timestamp_ms = (int(count * 1000 / video_fps) if args.video else
                                int((time.monotonic() - start) * 1000))
                count += 1
                result = extractor.process(raw_frame, timestamp_ms)
                buffer.push(result)
                now = time.monotonic()
                dt = now - last_wall
                last_wall = now
                if dt > 0:
                    fps = (0.90 * fps + 0.10 / dt) if fps else (1.0 / dt)

                ears = (eye_aspect_ratio(result.landmarks[0]),
                        eye_aspect_ratio(result.landmarks[1]))
                # One metrics calculation per ~0.5s, not on every camera frame.
                if result.timestamp_ms - last_calc_ms >= 500:
                    metrics = compute_rolling_metrics(buffer, calibration)
                    if metrics is not None:
                        metrics["face_missing_fraction"] = buffer.missing_fraction()
                    last_calc_ms = result.timestamp_ms

                if (csv_writer and metrics is not None and
                        result.timestamp_ms - last_log_ms >= 1000):
                    csv_writer.writerow(make_csv_row(metrics, calibration, result, ears, fps))
                    csv_file.flush()
                    last_log_ms = result.timestamp_ms

                frame = (cv2.cvtColor(raw_frame, cv2.COLOR_GRAY2BGR)
                         if raw_frame.ndim == 2 else raw_frame.copy())
                if result.face_found:
                    for eye in result.landmarks:
                        for x, y in eye:
                            cv2.circle(frame, (int(x), int(y)), 2, GREEN, -1)
                if not args.no_mirror:
                    frame = cv2.flip(frame, 1)
                fh, fw = frame.shape[:2]
                display_height = max(fh + GRAPH_H, 620)
                left = np.full((display_height, fw, 3), BG, dtype=np.uint8)
                left[:fh] = frame
                draw_graph(left, buffer, calibration)
                panel = draw_panel(display_height, metrics, calibration, buffer,
                                   result.face_found, ears[0], ears[1], fps, result.head_pose)
                cv2.imshow("Eye tracker | Rolling 60 s PERCLOS", np.hstack((left, panel)))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    savedir = REPO_ROOT / "recordings"
                    savedir.mkdir(parents=True, exist_ok=True)
                    out = savedir / f"buffer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
                    np.savez(out, **buffer.as_arrays())
                    print(f"Saved {len(buffer)} frames of numerical eye landmarks: {out}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if csv_file is not None:
            csv_file.close()


if __name__ == "__main__":
    main()
