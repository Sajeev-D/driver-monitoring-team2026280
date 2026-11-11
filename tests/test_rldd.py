"""RLDD labels, schema compatibility, discovery and resumed extraction contracts."""
from pathlib import Path
from types import SimpleNamespace
import csv
import json
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from eye_features import rldd
from eye_features.training_export import SYNTHETIC_FIELDS, training_fieldnames
from eye_features.calibration import EyeCalibration
from test_drozy import data, eye


def touch_video(root, subject, name, fold=1):
    p=root/f'Fold{fold}_part1'/f'Fold{fold}_part1'/subject/name
    p.parent.mkdir(parents=True,exist_ok=True)
    p.touch()
    return p


def arrays():
    lm,ts,found=data()
    lm[(ts>=10000)&(ts<20000)]=np.stack([eye(.05),eye(.05)])
    return {'landmarks':lm,'timestamps_ms':ts,'face_found':found,
            'head_pose':np.zeros((len(ts),3)), 'mouth_aspect_ratio':np.full(len(ts),.1)}


def test_nested_discovery_and_split_labels(tmp_path):
    for name in ('0.MOV','5.mp4','10_1.mov','10_2.MOV'):
        touch_video(tmp_path,'01',name)
    records=rldd.discover_rldd(tmp_path)
    assert [r['class_id'] for r in records]==[0,5,10,10]
    assert [r['drowsiness_label'] for r in records]==[0,None,1,1]
    assert len({r['session_id'] for r in records})==4
    assert all(r['fold']==1 and r['kss'] is None for r in records)
    extras=json.loads(records[1]['extra_labels_json'])
    assert (extras['kss_min'],extras['kss_max'])==(6,7)
    touch_video(tmp_path,'01','10.mp4')
    with pytest.raises(ValueError,match='whole and split'):
        rldd.discover_rldd(tmp_path)


def test_discovery_rejects_duplicates_unknown_labels_and_cross_fold(tmp_path):
    touch_video(tmp_path,'01','0.mov')
    duplicate=touch_video(tmp_path,'01','0.mp4')
    with pytest.raises(ValueError,match='Duplicate'):
        rldd.discover_rldd(tmp_path)
    duplicate.unlink()
    extra=touch_video(tmp_path,'01','8.mp4')
    with pytest.raises(ValueError,match='Unrecognized'):
        rldd.discover_rldd(tmp_path)
    extra.unlink()
    touch_video(tmp_path,'01','5.mov',fold=2)
    with pytest.raises(ValueError,match='multiple folds'):
        rldd.discover_rldd(tmp_path)


def test_missing_inventory_and_exact_schema(tmp_path):
    touch_video(tmp_path,'46','10.mov',fold=4)
    records=rldd.discover_rldd(tmp_path)
    inv=rldd.dataset_inventory(records)
    assert inv['missing_classes_by_subject']=={'46':[0,5]}
    assert inv['video_files']==1 and len(inv['missing_subjects'])==59
    fields=training_fieldnames()
    assert len(fields)==len(set(fields))==273
    assert fields[:22]==SYNTHETIC_FIELDS


def test_window_labels_blanks_and_real_normalization(tmp_path):
    touch_video(tmp_path,'01','5.mov')
    record=rldd.discover_rldd(tmp_path)[0]
    arr=arrays()
    profile={'sources':[{'session_id':record['session_id']}],'calibration_interval_s':[0,60]}
    calibration=EyeCalibration((.3,.3),(.05,.05))
    rows=rldd.rldd_windows(record,arr,calibration,profile,'test')
    assert len(rows)==2
    assert rows[0]['eye_openness']==pytest.approx(5/6)
    assert all(r['drowsiness_label'] is None and r['kss'] is None for r in rows)
    assert rows[0]['pvt_available'] is False
    assert np.isnan(rows[0]['pvt_mean_rt_ms'])
    assert rows[0]['psg_ecg_available'] is False
    assert rows[0].get('psg_ecg_rms') is None
    assert rows[0]['dropped_frame_indicator'] is None
    assert rows[0]['calibration_overlap']
    assert set(rows[0])<=set(training_fieldnames())


def test_mouth_extractor_pixel_coordinates_and_missing_face():
    # Synthetic landmark geometry: 10px width, each vertical pair separated 6px.
    face=[SimpleNamespace(x=0.,y=0.) for _ in range(478)]
    face[308].x=.1
    for a,b in ((82,87),(13,14),(312,317)):
        face[b].y=.03
    result=SimpleNamespace(face_landmarks=[face],facial_transformation_matrixes=[])
    extracted=rldd.RlddExtractor._to_frame_result(result,100,200,0)
    assert extracted.mouth_aspect_ratio==pytest.approx(.6)
    result.face_landmarks=[]
    assert np.isnan(rldd.RlddExtractor._to_frame_result(result,100,200,1).mouth_aspect_ratio)


def test_export_roundtrip_no_stale_rows_and_partial_debug(tmp_path,monkeypatch):
    root,out=tmp_path/'dataset',tmp_path/'output'
    for name in ('0.mov','5.mov','10.mov'):
        touch_video(root,'01',name)
    model=tmp_path/'model.task';model.touch()
    def fake_extract(record,model,output,max_seconds=None,timestamp_mode='pts'):
        arr=arrays()
        identity={'dataset_id':'UTA_RLDD','subject_id':record['subject_id']}
        cache=Path(output)/'cache';cache.mkdir(exist_ok=True)
        np.savez(cache/f'{record["session_id"]}.npz',**arr,identity=json.dumps(identity))
        return arr,identity
    monkeypatch.setattr(rldd,'extract_rldd',fake_extract)
    audit=rldd.export_rldd(root,out,model,max_seconds=70)
    assert audit['rows']==6 and audit['status']=='PARTIAL_DEBUG'
    with (out/'uta_rldd_indicators.csv').open() as f:
        reader=csv.DictReader(f);rows=list(reader)
        assert reader.fieldnames==training_fieldnames()
    assert {r['drowsiness_label'] for r in rows}=={'0','','1'}
    assert all(r['kss']==r['kss_rating']==r['psg_fz_theta_power']=='' for r in rows)
    assert all(json.loads(r['extra_labels_json'])['fold']==1 for r in rows)
    # A subset run rebuilds the output, rather than appending stale previous sessions.
    audit=rldd.export_rldd(root,out,model,sessions=['uta_01_10'],max_seconds=70)
    assert audit['rows']==2 and audit['binary_label_counts']=={'1':2}
    with (out/'uta_rldd_indicators.csv').open() as f:
        assert {r['session_id'] for r in csv.DictReader(f)}=={'uta_01_10'}


def test_debug_duration_validation_and_failed_sessions_saved(tmp_path,monkeypatch):
    with pytest.raises(ValueError,match='Debug prefix'):
        rldd.export_rldd(tmp_path,tmp_path/'out',tmp_path/'model',max_seconds=10)
    root=tmp_path/'dataset';touch_video(root,'01','0.mov')
    model=tmp_path/'model';model.touch()
    def broken(*args,**kwargs):
        raise ValueError('broken video')
    monkeypatch.setattr(rldd,'extract_rldd',broken)
    with pytest.raises(RuntimeError,match='1 video'):
        rldd.export_rldd(root,tmp_path/'out',model)
    audit=json.loads((tmp_path/'out/audit.json').read_text())
    assert audit['status']=='FAILED_SESSIONS' and audit['rows']==0
    assert audit['sessions'][0]['error']=='broken video'


@pytest.mark.parametrize('pts',[[0.,33.,100.],[0.,33.,33.]])
def test_extraction_uses_pts_and_reuses_only_matching_cache(tmp_path,monkeypatch,pts):
    import cv2
    video=touch_video(tmp_path/'dataset','01','0.mov')
    record=rldd.discover_rldd(tmp_path/'dataset')[0]
    model=tmp_path/'model';model.touch()
    output=tmp_path/'output'
    calls=[]
    class Capture:
        def __init__(self,path):
            self.index=0;calls.append(path)
        def isOpened(self):return True
        def get(self,prop):
            return {cv2.CAP_PROP_FRAME_COUNT:3,cv2.CAP_PROP_FPS:30,
                    cv2.CAP_PROP_POS_MSEC:pts[max(0,self.index-1)]}.get(prop,0)
        def read(self):
            self.index+=1
            return (True,np.zeros((4,4,3),dtype=np.uint8)) if self.index<=3 else (False,None)
        def release(self):pass
    class Extractor:
        def __init__(self,model):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def process(self,frame,ts):
            return rldd.RlddFrameResult(np.zeros((2,6,2)),np.zeros(3),ts,True,.1)
    monkeypatch.setattr(cv2,'VideoCapture',Capture)
    monkeypatch.setattr(rldd,'RlddExtractor',Extractor)
    if pts[-1]==pts[-2]:
        with pytest.raises(ValueError,match='invalid video PTS'):
            rldd.extract_rldd(record,model,output)
        assert not (output/'cache/uta_01_0.npz').exists()
    else:
        arr,_=rldd.extract_rldd(record,model,output)
        np.testing.assert_array_equal(arr['timestamps_ms'],pts)
        again,_=rldd.extract_rldd(record,model,output)
        assert len(calls)==1
        np.testing.assert_array_equal(arr['timestamps_ms'],again['timestamps_ms'])
        # An explicitly chosen FPS approximation cannot reuse the PTS cache.
        arr,_=rldd.extract_rldd(record,model,output,timestamp_mode='fps')
        assert len(calls)==2
        np.testing.assert_allclose(arr['timestamps_ms'],[0,1000/30,2000/30])
