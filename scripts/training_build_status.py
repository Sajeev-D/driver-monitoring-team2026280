#!/usr/bin/env python3
"""Report full-extraction progress without importing video-processing libraries."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
work=ROOT/'recordings/training_bundle_build'
p=json.loads((work/'progress.json').read_text())
clips,rows=0,0
for path in (work/'subjects').glob('*/audit.json'):
    audit=json.loads(path.read_text())
    clips+=sum(s['status']=='COMPLETE' for s in audit.get('sessions',[]))
    rows+=audit.get('rows',0)
resolved=[]
for subject in p['failed']:
    path=work/'subjects'/f'{int(subject):02d}'/'audit.json'
    if path.is_file() and json.loads(path.read_text()).get('status')=='COMPLETE_SELECTION':
        resolved.append(subject)
status={'status':p['status'],'drozy_rows':1946,'uta_subjects_total':p['subject_count'],
        'completed_subjects':p['completed']+resolved,'failed_subjects':[s for s in p['failed'] if s not in resolved],
        'retried_successfully':resolved,
        'uta_completed_videos':clips,'uta_windows_exported':rows,'active_subjects':p['active']}
out=ROOT/'data/processed/drowsiness/BUILD_STATUS.json'
out.parent.mkdir(parents=True,exist_ok=True)
temp=out.with_suffix('.tmp.json');temp.write_text(json.dumps(status,indent=2)+'\n');temp.replace(out)
print(json.dumps(status))
