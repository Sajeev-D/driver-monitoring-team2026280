#!/usr/bin/env python3
"""Run full exports and package portable training tables, with per-subject resume."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def write_json(path, data):
    temp=path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(data, indent=2)+'\n')
    temp.replace(path)


def package_tables(drozy, uta_sources, inventory, output):
    """Validate and concatenate identical schemas; preserve nulls and all classes."""
    from eye_features.training_export import training_fieldnames
    output.mkdir(parents=True, exist_ok=True)
    fields=training_fieldnames()
    reports={}
    all_keys=set()
    combined=output/'combined.csv'
    with combined.with_suffix('.tmp.csv').open('w',newline='') as both_handle:
        both=csv.DictWriter(both_handle,fieldnames=fields)
        both.writeheader()
        for name,sources in [('drozy',[drozy]),('uta_rldd',uta_sources)]:
            count, labels, subjects, sessions, classes=0,Counter(),set(),set(),Counter()
            time_sources=Counter()
            missing=Counter()
            overlap=0
            target=output/f'{name}.csv'
            with target.with_suffix('.tmp.csv').open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
                for source in sources:
                    with source.open(newline='') as source_handle:
                        reader=csv.DictReader(source_handle)
                        if reader.fieldnames!=fields:
                            raise ValueError(f'Schema mismatch: {source}')
                        for row in reader:
                            if None in row or any(v is None for v in row.values()):
                                raise ValueError(f'Malformed CSV row in {source}')
                            key=(row['dataset_id'],row['session_id'],row['window_start_s'])
                            if key in all_keys:
                                raise ValueError(f'Duplicate window: {key}')
                            all_keys.add(key)
                            if any(v.lower() in ('nan','inf','-inf') for v in row.values()):
                                raise ValueError(f'Nonfinite literal in {source}')
                            for field in ('eye_openness','perclos'):
                                if row[field] and not 0<=float(row[field])<=1:
                                    raise ValueError(f'{field} outside [0,1] in {source}')
                            count+=1;labels[row['drowsiness_label'] or 'unlabeled']+=1
                            time_sources[row['time_source']]+=1
                            subjects.add(row['subject_id']);sessions.add(row['session_id'])
                            overlap+=row['calibration_overlap']=='True'
                            missing.update(k for k,v in row.items() if v=='')
                            extras=json.loads(row['extra_labels_json'])
                            if 'uta_class_id' in extras:
                                classes[str(extras['uta_class_id'])]+=1
                            writer.writerow(row);both.writerow(row)
            target.with_suffix('.tmp.csv').replace(target)
            reports[name]={'rows':count,'participants':len(subjects),'sessions':len(sessions),
                           'binary_label_counts':dict(labels),'native_class_window_counts':dict(classes),
                           'time_source_window_counts':dict(time_sources),
                           'calibration_overlap_windows':overlap,'missing_counts':dict(missing),
                           'bytes':target.stat().st_size,
                           'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    combined.with_suffix('.tmp.csv').replace(combined)
    shutil.copyfile(ROOT/'src/eye_features/training_schema.json',output/'schema.json')
    manifest={'schema_version':1,'columns':len(fields),'window_s':60,'stride_s':10,
              'datasets':reports,'combined_rows':sum(r['rows'] for r in reports.values()),
              'combined_sha256':hashlib.sha256(combined.read_bytes()).hexdigest(),
              'uta_download':{k:v for k,v in inventory.items() if k!='records'},
              'source_urls':{'UTA_RLDD':'https://sites.google.com/view/utarldd/home',
                             'DROZY':'https://orbi.uliege.be/handle/2268/191620'},
              'notes':['Derived window features only; source videos and biosignals excluded.',
                       'UTA exact KSS unavailable; low-vigilance binary labels blank.',
                       'Use dataset_id plus subject_id for participant grouping.']}
    write_json(output/'manifest.json',manifest)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--uta-dataset',type=Path,required=True)
    p.add_argument('--drozy-dataset',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--work-dir',type=Path,default=ROOT/'recordings/training_bundle_build')
    p.add_argument('--output',type=Path,default=ROOT/'data/processed/drowsiness')
    p.add_argument('--package-only',action='store_true',help='Package an already successful run')
    args=p.parse_args()
    if not 1<=args.workers<=8:p.error('--workers must be 1..8')
    from eye_features.rldd import discover_rldd,dataset_inventory
    records=discover_rldd(args.uta_dataset)
    inventory=dataset_inventory(records)
    subjects=sorted({r['subject_id'] for r in records},key=int)
    work=args.work_dir;work.mkdir(parents=True,exist_ok=True)
    logs=work/'logs';logs.mkdir(exist_ok=True)
    env=dict(os.environ,MPLCONFIGDIR='/private/tmp/ece496-mpl',OMP_NUM_THREADS='1',
             OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    if not args.package_only:
        command=[sys.executable,str(ROOT/'scripts/export_drozy_training.py'),
                 '--dataset',str(args.drozy_dataset)]
        print('Refreshing DROZY features from existing caches',flush=True)
        with (logs/'drozy.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        queue=list(subjects);active={};done=[];failed=[]
        progress={'status':'RUNNING','subject_count':len(subjects),'workers':args.workers,
                  'inventory':{k:v for k,v in inventory.items() if k!='records'}}
        while queue or active:
            while queue and len(active)<args.workers:
                subject=queue.pop(0)
                out=work/'subjects'/f'{int(subject):02d}'
                command=[sys.executable,str(ROOT/'scripts/export_rldd_training.py'),
                         '--dataset',str(args.uta_dataset),'--subjects',subject,'--output',str(out)]
                log=(logs/f'uta_{int(subject):02d}.log').open('w')
                proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                active[subject]=(proc,log)
                print(f'Started UTA subject {subject} ({len(done)}/{len(subjects)} complete)',flush=True)
            for subject,(proc,log) in list(active.items()):
                code=proc.poll()
                if code is not None:
                    log.close();del active[subject]
                    (done if code==0 else failed).append(subject)
                    print(f'UTA subject {subject}: {"complete" if code==0 else "FAILED"}',flush=True)
            progress.update({'completed':done,'failed':failed,'active':list(active),'queued':queue})
            write_json(work/'progress.json',progress)
            if active:time.sleep(2)
        if failed:
            # A concurrent explicit retry may already have repaired a failed subject.
            unresolved=[]
            for subject in failed:
                path=work/'subjects'/f'{int(subject):02d}'/'audit.json'
                audit=json.loads(path.read_text()) if path.is_file() else {}
                expected={r['session_id'] for r in records if r['subject_id']==subject}
                if audit.get('status')=='COMPLETE_SELECTION' and {s['session_id'] for s in audit.get('sessions',[])}==expected:
                    done.append(subject)
                else:
                    unresolved.append(subject)
            failed=unresolved
            progress.update({'completed':done,'failed':failed})
            if failed:
                progress['status']='FAILED_SUBJECTS';write_json(work/'progress.json',progress)
                raise RuntimeError(f'UTA failed subjects {failed}; successful caches retained, see {logs}')
        progress['status']='COMPLETE';write_json(work/'progress.json',progress)
    sources=[]
    for subject in subjects:
        out=work/'subjects'/f'{int(subject):02d}'
        audit=json.loads((out/'audit.json').read_text())
        expected={r['session_id'] for r in records if r['subject_id']==subject}
        if audit['status']!='COMPLETE_SELECTION' or {s['session_id'] for s in audit['sessions']}!=expected:
            raise ValueError(f'Subject {subject} export is incomplete or stale')
        sources.append(out/'uta_rldd_indicators.csv')
    manifest=package_tables(ROOT/'recordings/drozy_training/drozy_indicators.csv',sources,inventory,args.output)
    manifest['timestamp_repairs']=[]
    manifest['frame_count_metadata_corrections']=[]
    for subject in subjects:
        audit=json.loads((work/'subjects'/f'{int(subject):02d}'/'audit.json').read_text())
        for session in audit['sessions']:
            identity=session.get('extraction',{})
            if identity.get('timestamp_repairs'):
                manifest['timestamp_repairs'].append({'session_id':session['session_id'],
                    'method':identity['timestamp_repair_method'],'frames':identity['timestamp_repairs']})
            if identity.get('frame_count_metadata_correction'):
                manifest['frame_count_metadata_corrections'].append({'session_id':session['session_id'],
                    **identity['frame_count_metadata_correction']})
    write_json(args.output/'manifest.json',manifest)
    write_json(args.output/'BUILD_STATUS.json',{'status':'COMPLETE_BUNDLE',
        'drozy_recordings':manifest['datasets']['drozy']['sessions'],
        'uta_video_files':manifest['datasets']['uta_rldd']['sessions'],
        'combined_rows':manifest['combined_rows'],'columns':manifest['columns']})
    print(f'Packaged {manifest["combined_rows"]} windows in {args.output}',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
