"""Small local audit images and plots using the existing OpenCV dependency."""

from pathlib import Path

import cv2
import numpy as np

from .offline import write_csv


def _text(image, text, x, y, color=(235, 235, 235), scale=0.45):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def export_qa(video: Path, directory: Path, arrays: dict, timeline):
    """Candidates are examples for review, never automatically closed references."""
    directory.mkdir(parents=True, exist_ok=True)
    valid = np.flatnonzero(np.all(timeline.valid, axis=1))
    candidates = []
    if len(valid):
        mean_ear = np.mean(timeline.ear[valid], axis=1)
        for label, quantile in (("low_01", .01), ("low_05", .05), ("low_10", .10),
                                ("high_90", .90), ("high_95", .95), ("high_99", .99)):
            target = np.quantile(mean_ear, quantile)
            index = int(valid[np.argmin(abs(mean_ear - target))])
            candidates.append((label, index))
    cap = cv2.VideoCapture(str(video))
    tiles, candidate_rows = [], []
    try:
        for label, index in candidates:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Could not read QA frame {index} from {video}")
            annotated = frame.copy()
            points = arrays["landmarks"][index]
            for eye, color in zip(points, ((80, 255, 80), (255, 200, 70))):
                for x, y in eye:
                    cv2.circle(annotated, (round(float(x)), round(float(y))), 1, color, -1)
            # Preserve an unannotated image for actual eyelid inspection.
            filename = f"{label}_frame_{index}.png"
            if not cv2.imwrite(str(directory / filename), frame):
                raise OSError(f"Could not write {directory / filename}")
            tile = np.full((390, 512, 3), 25, dtype=np.uint8)
            tile[35:255] = cv2.resize(annotated, (512, 220))
            _text(tile, f"{video.stem} {label} frame {index} t={timeline.timestamps_s[index]:.3f}s", 8, 19)
            _text(tile, f"EAR right={timeline.ear[index,0]:.3f} left={timeline.ear[index,1]:.3f}", 8, 279)
            # Raw eye crop, enlarged with no landmarks obscuring the eyelids.
            low = np.floor(points.reshape(-1, 2).min(axis=0) - [12, 14]).astype(int)
            high = np.ceil(points.reshape(-1, 2).max(axis=0) + [12, 14]).astype(int)
            x0, y0 = np.maximum(low, 0)
            x1, y1 = np.minimum(high, [frame.shape[1], frame.shape[0]])
            if x1 > x0 and y1 > y0:
                tile[290:390] = cv2.resize(frame[y0:y1, x0:x1], (512, 100))
            tiles.append(tile)
            candidate_rows.append({"session_id": video.stem, "candidate": label,
                                   "frame_index": index, "timestamp_ms": int(arrays["timestamps_ms"][index]),
                                   "ear_right": float(timeline.ear[index, 0]), "ear_left": float(timeline.ear[index, 1]),
                                   "image": filename, "verified_open_or_closed": False})
    finally:
        cap.release()
    if tiles:
        montage = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
        cv2.imwrite(str(directory / "reference_candidates.png"), montage)
    write_csv(directory / "reference_candidates.csv", candidate_rows,
              ["session_id", "candidate", "frame_index", "timestamp_ms", "ear_right", "ear_left", "image", "verified_open_or_closed"])
    plot = np.full((550, 1400, 3), 25, dtype=np.uint8)
    start, end = timeline.timestamps_s[[0, -1]]
    y_max = max(.4, float(np.nanquantile(timeline.ear, .995)) * 1.1) if len(valid) else .4
    y_max = min(y_max, 1.0)
    _text(plot, f"{video.stem}: MediaPipe EAR; green=right, blue=left; red=invalid; references require review", 15, 22, scale=.6)
    for eye in range(2):
        top, height = 60 + eye * 225, 175
        cv2.rectangle(plot, (70, top), (1360, top + height), (100, 100, 100), 1)
        _text(plot, ("RIGHT", "LEFT")[eye], 8, top + 80)
        for tick in np.linspace(0, y_max, 5):
            y = top + height - round(tick / y_max * height)
            cv2.line(plot, (70, y), (1360, y), (50, 50, 50))
            _text(plot, f"{tick:.2f}", 28, y + 4, scale=.35)
        # Per-pixel min/max envelope preserves short blinks at full-session scale.
        bins = np.minimum(1289, ((timeline.timestamps_s - start) / (end - start) * 1290).astype(int))
        for pixel in np.unique(bins):
            mask = bins == pixel
            values = timeline.ear[mask & timeline.valid[:, eye], eye]
            if len(values):
                a, b = np.clip([values.min(), values.max()], 0, y_max)
                color = ((80, 240, 80), (255, 190, 80))[eye]
                cv2.line(plot, (70 + pixel, top + height - round(b / y_max * height)),
                         (70 + pixel, top + height - round(a / y_max * height)), color, 1)
            if not np.all(timeline.valid[mask, eye]):
                cv2.line(plot, (70 + pixel, top + height + 5), (70 + pixel, top + height + 9), (70, 70, 255))
        for tick in np.linspace(start, end, 11):
            x = 70 + round((tick - start) / (end - start) * 1290)
            _text(plot, f"{tick:.0f}s", x - 12, top + height + 25, scale=.35)
    cv2.imwrite(str(directory / "ear_timeline.png"), plot)
