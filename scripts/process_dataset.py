#!/usr/bin/env python3
"""Process videos from any dataset using a manifest and the common CSV schema."""

import argparse
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "recordings" / "dataset")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "face_landmarker.task")
    parser.add_argument("--sessions", nargs="+", help="Optional subset; default all available manifest videos")
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--window-s", type=float, default=60)
    parser.add_argument("--stride-s", type=float, default=10)
    parser.add_argument("--pvt-lapse-ms", type=float, default=500)
    parser.add_argument("--max-seconds", type=float, help="Debug prefix only")
    args = parser.parse_args()
    for name in ("window_s", "stride_s", "pvt_lapse_ms", "max_seconds"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"{name} must be finite and positive")
    from eye_features.dataset_manifest import read_manifest
    from eye_features.offline import run_records
    try:
        run_records(read_manifest(args.manifest), args.output, args.model, args.sessions,
                    args.calibration_dir, args.window_s, args.stride_s,
                    args.max_seconds, args.pvt_lapse_ms)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Dataset processing failed: {error}", file=sys.stderr)
        return 1
    print(f"Results: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
