#!/usr/bin/env python3
"""Create a UTA-RLDD CSV matching DROZY's fixed 273-column training schema."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'recordings/uta_rldd_training')
    p.add_argument('--model',type=Path,default=ROOT/'models/face_landmarker.task')
    p.add_argument('--subjects',nargs='+',help='Optional subject IDs, e.g. 01 02')
    p.add_argument('--sessions',nargs='+',help='Optional session IDs from inventory.json')
    p.add_argument('--calibration-s',type=float,default=60)
    p.add_argument('--window-s',type=float,default=60)
    p.add_argument('--stride-s',type=float,default=10)
    p.add_argument('--max-seconds',type=float,help='Debug prefix; full run omits this flag')
    p.add_argument('--timestamp-mode',choices=['pts','fps'],default='pts')
    p.add_argument('--inventory-only',action='store_true',help='Discover videos without extracting frames')
    args=p.parse_args()
    from eye_features.rldd import export_rldd
    try:
        audit=export_rldd(args.dataset,args.output,args.model,args.subjects,args.sessions,
                          args.max_seconds,args.calibration_s,args.window_s,args.stride_s,
                          args.timestamp_mode,args.inventory_only)
    except (ValueError,OSError,RuntimeError) as error:
        print(f'UTA-RLDD export failed: {error}',file=sys.stderr)
        return 1
    if args.inventory_only:
        print(f'Found {audit["video_files"]} videos from {audit["participants"]} subjects')
    else:
        print(f'Wrote {audit["rows"]} rows x {audit["columns"]} columns to {args.output}')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
