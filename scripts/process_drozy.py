#!/usr/bin/env python3
"""Run MediaPipe over DROZY recordings and export auditable window features."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Local DROZY directory")
    parser.add_argument("--output", type=Path, default=ROOT / "recordings" / "drozy_pilot")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "face_landmarker.task")
    parser.add_argument("--sessions", nargs="+", default=["1-1", "1-3"], help="Subject-test IDs (default: 1-1 1-3)")
    parser.add_argument("--all", action="store_true", help="Process all available recordings")
    parser.add_argument("--calibration-dir", type=Path, help="Reviewed subject_N.json profiles; no webcam profile fallback")
    parser.add_argument("--window-s", type=float, default=60.0)
    parser.add_argument("--stride-s", type=float, default=10.0)
    parser.add_argument("--pvt-lapse-ms", type=float, default=500.0, help="Explicit analysis convention, not an official DROZY binary label")
    parser.add_argument("--max-seconds", type=float, help="Debug only: process a prefix; manifest marks PARTIAL_DEBUG")
    args = parser.parse_args()
    import math
    for name in ("window_s", "stride_s", "pvt_lapse_ms", "max_seconds"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    from eye_features.drozy import run_batch
    try:
        run_batch(args.dataset, args.output, args.model, None if args.all else args.sessions,
                  args.calibration_dir, args.window_s, args.stride_s,
                  args.max_seconds, args.pvt_lapse_ms)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"DROZY processing failed: {error}", file=sys.stderr)
        return 1
    print(f"Results: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
