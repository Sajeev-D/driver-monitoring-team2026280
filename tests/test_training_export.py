"""Independent numerical checks for the multimodal CSV exporter."""
from pathlib import Path
import sys
import csv
import json

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from eye_features.training_export import bandpowers, completed_runs, read_edf, psg_features


def test_welch_known_sine_power():
    fs=512
    t=np.arange(60*fs)/fs
    result=bandpowers(3*np.sin(2*np.pi*6*t),fs)
    assert result['theta_power']==pytest.approx(4.5,rel=.001)
    assert result['theta_relative']>.999


def test_candidate_events_exclude_censoring_and_tracking_gaps():
    times=np.arange(9,dtype=float)
    valid=np.ones(9,dtype=bool)
    mask=np.array([True,False,True,True,False,True,True,False,True])
    assert completed_runs(mask,valid,times,1)==[(2.,4.,2.),(5.,7.,2.)]
    valid[3]=False
    assert completed_runs(mask,valid,times,1)==[(5.,7.,2.)]
    times[6:]+=10
    assert completed_runs(mask,valid,times,1)==[]


def test_edf_scaling_channel_layout_and_payload_validation(tmp_path):
    def field(value,width):
        return str(value).ljust(width).encode('ascii')
    path=tmp_path/'test.edf'
    h=(field('0',8)+field('patient',80)+field('recording',80)+field('01.01.20',8)
       +field('12.00.00',8)+field(768,8)+field('',44)+field(2,8)+field(1,8)+field(2,4))
    sh=b''
    for values,width in [(['Fz','ECG'],16),(['',''],80),(['uV','uV'],8),
                         ([-10,-20],8),([10,20],8),([-100,-100],8),([100,100],8),
                         (['',''],80),([2,1],8),(['',''],32)]:
        sh+=b''.join(field(v,width) for v in values)
    payload=np.array([-100,0,100,100,50,-100],dtype='<i2').tobytes()
    path.write_bytes(h+sh+payload)
    start,signals=read_edf(path)
    assert start.isoformat()=='2020-01-01T12:00:00'
    np.testing.assert_allclose(signals['Fz']['values'],[-10,0,10,5])
    np.testing.assert_allclose(signals['ECG']['values'],[20,-20])
    assert signals['Fz']['fs']==2
    assert psg_features(signals,0,0,3)['psg_window_complete'] is False
    path.write_bytes(h+sh+payload[:-2])
    with pytest.raises(ValueError,match='payload length'):
        read_edf(path)


def test_export_preserves_labels_and_synthetic_schema(tmp_path, monkeypatch):
    from datetime import datetime
    from eye_features import training_export as module
    from test_drozy import data, eye
    dataset, processed, output = (tmp_path/name for name in ('dataset','processed','output'))
    for folder in ('timestamps','annotations-auto','interpIndices','pvt-rt'):
        (dataset/folder).mkdir(parents=True)
    (processed/'cache').mkdir(parents=True)
    lm,ts,found=data()
    lm[(ts>=10000)&(ts<20000)]=np.stack([eye(.05),eye(.05)])
    np.savez(processed/'cache/1-1.npz',landmarks=lm,timestamps_ms=ts,face_found=found,
             head_pose=np.zeros((len(ts),3)),
             identity=json.dumps({'subject_id':'1','dataset_id':'DROZY'}))
    stamp=np.zeros((len(ts),8));stamp[:,7]=ts
    np.savetxt(dataset/'timestamps/1-1.txt',stamp)
    native=np.tile(np.arange(136),(len(ts),1))
    np.savetxt(dataset/'annotations-auto/1-1-s2.txt',native)
    np.savetxt(dataset/'interpIndices/1-1.txt',np.tile([1,0,1],(18000,1)))
    (dataset/'pvt-rt/1-1.csv').write_text('2020-01-01_12.00.00.000\n')
    record={'subject_id':'1','session_id':'1-1','test_id':'1','kss':7,
            'dataset_id':'DROZY','video_available':True,'kss_label_scope':'session'}
    monkeypatch.setattr(module,'drozy_records',lambda root:[record])
    monkeypatch.setattr(module,'load_pvt',lambda rec:[])
    monkeypatch.setattr(module,'read_edf',lambda path:(datetime(2020,1,1,12),{}))
    audit=module.export_training(dataset,processed,output)
    assert json.loads((output/'audit.json').read_text())['rows']==2
    with (output/'drozy_indicators.csv').open() as f:
        reader=csv.DictReader(f)
        rows=list(reader)
        assert reader.fieldnames[:22]==module.SYNTHETIC_FIELDS
    assert audit['binary_label_counts']=={'0':0,'1':2}
    assert rows[0]['kss_rating']=='7'
    assert rows[0]['sleep_duration']==''
    assert rows[0]['perclos']==rows[0]['perclos_combined']
    assert rows[0]['yawning_rate_per_min']==rows[0]['mouth_open_candidate_rate_per_min']
    assert rows[0]['calibration_overlap']=='True'
