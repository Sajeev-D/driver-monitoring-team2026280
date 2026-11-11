#!/usr/bin/env python3
"""Save per-subject references for any dataset from reviewed frames or min/max EAR in a cached interval."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--subject", required=True, help="Subject ID matching the manifest")
    parser.add_argument("--method", choices=["reviewed", "minmax"], default="reviewed")
    parser.add_argument("--session", help="Calibration recording for minmax mode")
    parser.add_argument("--start-s", type=float, help="Start of minmax calibration interval")
    parser.add_argument("--end-s", type=float, help="End of minmax calibration interval")
    parser.add_argument("--open-session")
    parser.add_argument("--open-frame", type=int, help="Zero-based decoded video index")
    parser.add_argument("--closed-session")
    parser.add_argument("--closed-frame", type=int, help="Zero-based decoded video index")
    parser.add_argument("--reviewer")
    parser.add_argument("--confirm-eyes-open-and-closed", action="store_true",
                        help="Confirm visual review: both eyes open in open frame, suitably closed in closed frame")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from eye_features.offline import calibration_from_reviewed_frames, calibration_from_minmax
    try:
        if args.method == "minmax":
            if args.session is None or args.start_s is None or args.end_s is None:
                parser.error("minmax requires --session, --start-s and --end-s")
            profile = calibration_from_minmax(args.cache_dir, args.subject, args.session,
                                             args.start_s, args.end_s)
        else:
            if not args.confirm_eyes_open_and_closed:
                parser.error("Visual review required: --confirm-eyes-open-and-closed")
            if any(value is None for value in (args.open_session, args.open_frame,
                    args.closed_session, args.closed_frame, args.reviewer)):
                parser.error("reviewed requires open/closed sessions and frames, and --reviewer")
            profile = calibration_from_reviewed_frames(args.cache_dir, args.subject,
                                                      args.open_session, args.open_frame,
                                                      args.closed_session, args.closed_frame,
                                                      args.reviewer)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}; rerun process_drozy.py with --calibration-dir {args.output.parent}")


if __name__ == "__main__":
    main()
