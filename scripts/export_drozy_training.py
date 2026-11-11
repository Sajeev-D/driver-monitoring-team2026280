#!/usr/bin/env python3
"""Export multimodal training features from DROZY and existing extraction caches."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--processed',type=Path,default=ROOT/'recordings/drozy_all')
    p.add_argument('--output',type=Path,default=ROOT/'recordings/drozy_training')
    p.add_argument('--calibration-s',type=float,default=60)
    p.add_argument('--kss-threshold',type=int,default=7)
    args=p.parse_args()
    from eye_features.training_export import export_training
    audit=export_training(args.dataset,args.processed,args.output,args.calibration_s,args.kss_threshold)
    print(f'Wrote {audit["rows"]} rows x {audit["columns"]} columns to {args.output}')

if __name__=='__main__':
    main()
