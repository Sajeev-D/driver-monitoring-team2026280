#!/usr/bin/env python3
"""Retry a subject with isolated duplicate PTS interpolated between real neighbours.

Reject backward timestamps, terminal duplicates and broad timing damage.
Record every corrected frame and both timestamps in cache/audit metadata.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def repair_duplicates(values):
    values=np.asarray(values,dtype=float)
    if len(values)<2 or not np.isfinite(values).all() or np.any(np.diff(values)<0):
        raise ValueError('Only finite monotone PTS with isolated duplicates can be repaired')
    bad=np.flatnonzero(np.diff(values)==0)+1
    if len(bad)>max(1,len(values)//1000):
        raise ValueError('Too many duplicate timestamps for isolated interpolation')
    fixed=values.copy();repairs=[]
    for i in bad:
        if i+1>=len(values) or values[i+1]<=values[i] or (i>1 and values[i-1]==values[i-2]):
            raise ValueError('Only isolated interior duplicate timestamps can be repaired')
        fixed[i]=(values[i-1]+values[i+1])/2
        repairs.append({'frame_index':int(i),'original_ms':float(values[i]),'corrected_ms':float(fixed[i])})
    return fixed,repairs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--subject',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-confirmed-frame-count-correction',action='store_true',
                   help='Accept a container count mismatch only when FFmpeg independently confirms the decoded count')
    args=p.parse_args()
    import cv2
    from eye_features import rldd
    records=[r for r in rldd.discover_rldd(args.dataset) if r['subject_id']==str(int(args.subject))]
    if not records:raise ValueError('Unknown subject')
    original_capture=cv2.VideoCapture
    original_fingerprint=rldd.fingerprint
    original_windows=rldd.rldd_windows
    timing={};counts={}
    for record in records:
        cap=original_capture(record['video_path']);times=[]
        declared=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        try:
            while cap.grab():
                times.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        finally:cap.release()
        fixed,repairs=repair_duplicates(times)
        if declared!=len(times):
            if not args.allow_confirmed_frame_count_correction:
                raise ValueError('Container/decoded frame count differs; independent confirmation required')
            result=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
                '-show_entries','stream=nb_read_frames','-of','json',record['video_path']],
                capture_output=True,text=True,check=True)
            independent=int(json.loads(result.stdout)['streams'][0]['nb_read_frames'])
            if result.stderr.strip() or independent!=len(times):
                raise ValueError(f'FFmpeg did not independently confirm clean decoded count: {result.stderr}')
            counts[str(Path(record['video_path']).resolve())]={'container_frames':declared,
                'decoded_frames':len(times),'ffprobe_frames':independent}
            print(record['session_id'],'container count corrected:',declared,'->',independent,flush=True)
        if repairs:
            timing[str(Path(record['video_path']).resolve())]=(fixed,repairs)
        print(record['session_id'],len(times),'frames;',len(repairs),'duplicate PTS corrections',flush=True)
    if not timing and not counts:raise ValueError('No metadata repair needed; use the standard exporter')
    class Capture:
        def __init__(self,path,*a,**kw):
            self.inner=original_capture(path,*a,**kw);self.index=-1
            self.times=timing.get(str(Path(path).resolve()),(None,None))[0]
            self.count=counts.get(str(Path(path).resolve()))
        def read(self):
            ok,frame=self.inner.read()
            if ok:self.index+=1
            return ok,frame
        def get(self,prop):
            if prop==cv2.CAP_PROP_POS_MSEC and self.times is not None:
                return float(self.times[self.index])
            if prop==cv2.CAP_PROP_FRAME_COUNT and self.count is not None:
                return self.count['decoded_frames']
            return self.inner.get(prop)
        def __getattr__(self,name):return getattr(self.inner,name)
    def fingerprint(video,timestamps,model):
        identity=original_fingerprint(video,timestamps,model)
        item=timing.get(str(Path(video).resolve()))
        if item:
            identity['timestamp_repairs']=item[1]
            identity['timestamp_repair_method']='isolated_duplicate_pts_neighbour_midpoint'
            identity['timestamp_repair_code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        count=counts.get(str(Path(video).resolve()))
        if count:
            identity['frame_count_metadata_correction']=count
            identity['frame_count_repair_code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return identity
    def windows(record,*a,**kw):
        rows=original_windows(record,*a,**kw)
        if str(Path(record['video_path']).resolve()) in timing:
            for row in rows:row['time_source']='video_pts_isolated_duplicate_interpolated'
        elif str(Path(record['video_path']).resolve()) in counts:
            for row in rows:row['time_source']='video_pts_confirmed_container_count_correction'
        return rows
    cv2.VideoCapture=Capture;rldd.fingerprint=fingerprint;rldd.rldd_windows=windows
    try:
        result=rldd.export_rldd(args.dataset,args.output,ROOT/'models/face_landmarker.task',subjects=[args.subject])
    finally:
        cv2.VideoCapture=original_capture;rldd.fingerprint=original_fingerprint;rldd.rldd_windows=original_windows
    print('Repaired subject complete:',result['rows'],'windows',flush=True)


if __name__=='__main__':main()
