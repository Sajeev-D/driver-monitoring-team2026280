"""Training bundle keeps identical schemas, distinct participants and null labels."""
from pathlib import Path
import csv
import importlib.util
import json
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from eye_features.training_export import training_fieldnames
spec=importlib.util.spec_from_file_location('training_bundle_builder',ROOT/'scripts/build_training_bundle.py')
builder=importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def source(path,dataset,session,label,extras='{}'):
    row={k:'' for k in training_fieldnames()}
    row.update({'dataset_id':dataset,'subject_id':'1','session_id':session,
                'window_start_s':'0.0','drowsiness_label':label,'extra_labels_json':extras,
                'eye_openness':'0.8','perclos':'0.1','calibration_overlap':'True'})
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=training_fieldnames());writer.writeheader();writer.writerow(row)


def test_bundle_preserves_all_rows_and_missing_labels(tmp_path):
    drozy,uta=tmp_path/'drozy.csv',tmp_path/'uta.csv'
    source(drozy,'DROZY','1-1','0')
    source(uta,'UTA_RLDD','uta_01_5','',json.dumps({'uta_class_id':5}))
    out=tmp_path/'bundle'
    manifest=builder.package_tables(drozy,[uta],{'video_files':1,'records':[]},out)
    assert manifest['combined_rows']==2
    assert manifest['datasets']['uta_rldd']['binary_label_counts']=={'unlabeled':1}
    assert manifest['datasets']['uta_rldd']['native_class_window_counts']=={'5':1}
    assert json.loads((out/'schema.json').read_text())==training_fieldnames()
    assert 'records' not in manifest['uta_download']
    with (out/'combined.csv').open() as f:
        reader=csv.DictReader(f);rows=list(reader)
        assert reader.fieldnames==training_fieldnames()
    assert {r['dataset_id'] for r in rows}=={'DROZY','UTA_RLDD'}
    assert rows[1]['drowsiness_label']==''


def test_bundle_rejects_duplicate_windows_and_mismatched_schema(tmp_path):
    drozy=tmp_path/'drozy.csv';source(drozy,'DROZY','1-1','0')
    with pytest.raises(ValueError,match='Duplicate window'):
        builder.package_tables(drozy,[drozy],{},tmp_path/'duplicate')
    wrong=tmp_path/'wrong.csv';wrong.write_text('foo,bar\n1,2\n')
    with pytest.raises(ValueError,match='Schema mismatch'):
        builder.package_tables(drozy,[wrong],{},tmp_path/'wrong')


def test_isolated_duplicate_pts_repair_preserves_other_times():
    import numpy as np
    spec=importlib.util.spec_from_file_location('repair_pts',ROOT/'scripts/repair_rldd_duplicate_pts.py')
    repair=importlib.util.module_from_spec(spec);spec.loader.exec_module(repair)
    original=np.array([0.,33.,66.,66.,100.])
    fixed,changes=repair.repair_duplicates(original)
    np.testing.assert_array_equal(fixed,[0.,33.,66.,83.,100.])
    np.testing.assert_array_equal(original,[0.,33.,66.,66.,100.])
    assert changes==[{'frame_index':3,'original_ms':66.,'corrected_ms':83.}]
    for bad in ([0,33,20],[0,33,33],[0,0,0,33],[0,float('nan'),33]):
        with pytest.raises(ValueError):repair.repair_duplicates(bad)
