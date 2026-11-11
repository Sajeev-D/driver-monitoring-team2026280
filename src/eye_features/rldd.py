"""UTA-RLDD discovery, resumable RGB extraction and common-schema CSV export."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json
import re
import time

import numpy as np

from .calibration import EyeCalibration
from .extractor import EyeFeatureExtractor, FrameResult
from .offline import (calibration_from_minmax, make_timeline, window_metrics,
                      write_csv, fingerprint)
from .training_export import add_video_indicators, training_fieldnames

SOURCE_URL = 'https://sites.google.com/view/utarldd/home'
CLASSES = {0: ('alert', 1, 3), 5: ('low_vigilance', 6, 7), 10: ('drowsy', 8, 9)}
VIDEO_SUFFIXES = {'.mov', '.mp4', '.avi', '.mkv', '.m4v'}


def discover_rldd(root):
    """Accept official subject folders and nested Kaggle FoldN_partN folders."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f'Dataset directory does not exist: {root}')
    records, keys, subject_folds = [], set(), {}
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.name.startswith('._') or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        match = re.fullmatch(r'(0|5|10)(?:_(\d+))?', path.stem)
        if not match or not path.parent.name.isdecimal():
            raise ValueError(f'Unrecognized RLDD video name/subject folder: {path}')
        subject = str(int(path.parent.name))
        if not 1 <= int(subject) <= 60:
            raise ValueError(f'Invalid RLDD subject: {path.parent.name}')
        label, segment = int(match[1]), match[2]
        key = (subject, label, int(segment) if segment else None)
        if key in keys:
            raise ValueError(f'Duplicate subject/class/segment video: {path}')
        keys.add(key)
        folds = {int(m[1]) for part in path.relative_to(root).parts[:-1]
                 if (m := re.fullmatch(r'Fold(\d+)(?:_part\d+)?', part, flags=re.I))}
        if len(folds) > 1 or any(not 1 <= f <= 5 for f in folds):
            raise ValueError(f'Ambiguous/invalid fold directory: {path}')
        fold = next(iter(folds), None)
        if subject in subject_folds and subject_folds[subject] != fold:
            raise ValueError(f'Subject {subject} appears in multiple folds')
        subject_folds[subject] = fold
        session = f'uta_{int(subject):02d}_{label}' + (f'_part{int(segment)}' if segment else '')
        name, low, high = CLASSES[label]
        extras = {'uta_class_id': label, 'uta_class_name': name, 'kss_min': low,
                  'kss_max': high, 'fold': fold, 'segment': int(segment) if segment else None}
        records.append({'dataset_id': 'UTA_RLDD', 'subject_id': subject, 'session_id': session,
                        'test_id': path.stem, 'video_path': str(path), 'class_id': label,
                        'fold': fold, 'segment': int(segment) if segment else None,
                        'kss': None, 'kss_label_scope': None,
                        'drowsiness_label': {0: 0, 5: None, 10: 1}[label],
                        'label_scope': 'session', 'label_source': SOURCE_URL,
                        'extra_labels_json': json.dumps(extras, sort_keys=True),
                        'time_source': 'video_pts_zero_origin'})
    if not records:
        raise ValueError(f'No RLDD videos found under {root}')
    # A whole clip plus split copies could duplicate the same observations.
    for subject, label, segment in keys:
        if segment is not None and (subject, label, None) in keys:
            raise ValueError(f'Subject {subject}, class {label}: whole and split videos both present')
    return sorted(records, key=lambda r: (int(r['subject_id']), r['class_id'], r['segment'] or 0))


def dataset_inventory(records):
    subjects = sorted({r['subject_id'] for r in records}, key=int)
    missing = {subject: [c for c in CLASSES if not any(r['subject_id'] == subject
               and r['class_id'] == c for r in records)] for subject in subjects}
    return {'video_files': len(records), 'participants': len(subjects),
            'class_file_counts': dict(Counter(str(r['class_id']) for r in records)),
            'missing_subjects': [str(s) for s in range(1, 61) if str(s) not in subjects],
            'missing_classes_by_subject': {s: v for s, v in missing.items() if v},
            'folds': sorted({r['fold'] for r in records if r['fold'] is not None}),
            'source_url': SOURCE_URL, 'records': records}


@dataclass
class RlddFrameResult(FrameResult):
    mouth_aspect_ratio: float = np.nan


class RlddExtractor(EyeFeatureExtractor):
    @staticmethod
    def _to_frame_result(result, width, height, timestamp_ms):
        base = EyeFeatureExtractor._to_frame_result(result, width, height, timestamp_ms)
        mar = np.nan
        if result.face_landmarks:
            face = result.face_landmarks[0]
            def distance(a, b):
                return np.hypot((face[a].x-face[b].x)*width, (face[a].y-face[b].y)*height)
            mouth_width = distance(78, 308)
            if mouth_width > 0:
                mar = sum(distance(a, b) for a, b in ((82,87),(13,14),(312,317))) / (3*mouth_width)
        return RlddFrameResult(base.landmarks, base.head_pose, base.timestamp_ms,
                               base.face_found, float(mar))


def extract_rldd(record, model, output, max_seconds=None, timestamp_mode='pts'):
    """Decode every frame; cache identity includes video/model/code/options.

    PTS preserves variable frame spacing. Explicit FPS mode is an opt-in
    approximation for broken container timestamps and is recorded as such.
    """
    import cv2
    if timestamp_mode not in {'pts', 'fps'}:
        raise ValueError('timestamp_mode must be pts or fps')
    video, model, output = Path(record['video_path']), Path(model), Path(output)
    identity = fingerprint(video, None, model)
    identity.update({'dataset_id': 'UTA_RLDD', 'subject_id': record['subject_id'],
                     'session_id': record['session_id'], 'timestamp_mode': timestamp_mode,
                     'max_seconds': max_seconds, 'rldd_version': 1,
                     'rldd_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    cache_path = output/'cache'/f'{record["session_id"]}.npz'
    names = ('landmarks','timestamps_ms','face_found','head_pose','mouth_aspect_ratio')
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cache:
            if json.loads(str(cache['identity'])) == identity:
                print(f'{record["session_id"]}: reusing validated cache', flush=True)
                return {k: cache[k].copy() for k in names}, identity
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f'Cannot open {video}')
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    values = {k: [] for k in names}
    first_pts, truncated = None, False
    try:
        if timestamp_mode == 'fps' and (not np.isfinite(fps) or fps <= 0):
            raise ValueError('Cannot use FPS mode without finite positive container FPS')
        with RlddExtractor(model) as extractor:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                index = len(values['timestamps_ms'])
                pts = index*1000/fps if timestamp_mode == 'fps' else float(cap.get(cv2.CAP_PROP_POS_MSEC))
                if first_pts is None:
                    first_pts = pts
                ts = pts-first_pts
                if not np.isfinite(ts) or ts < 0 or (index and ts <= values['timestamps_ms'][-1]):
                    raise ValueError(f'{record["session_id"]}: invalid video PTS at frame {index}; '
                                     'use --timestamp-mode fps only if constant FPS is acceptable')
                if max_seconds is not None and ts > max_seconds*1000:
                    truncated = True
                    break
                result = extractor.process(frame, round(ts))
                values['landmarks'].append(result.landmarks)
                values['head_pose'].append(result.head_pose)
                values['face_found'].append(result.face_found)
                values['mouth_aspect_ratio'].append(result.mouth_aspect_ratio)
                values['timestamps_ms'].append(ts)
                if index % 1500 == 0:
                    print(f'{record["session_id"]}: frame {index+1}/{expected}, t={ts/1000:.1f}s', flush=True)
        count = len(values['timestamps_ms'])
        if count < 2 or (not truncated and expected > 0 and count != expected):
            raise ValueError(f'{record["session_id"]}: decoded {count}, container reports {expected} frames')
    finally:
        cap.release()
    current = video.stat()
    if current.st_size != identity['video_size'] or current.st_mtime_ns != identity['video_mtime_ns']:
        raise ValueError(f'{record["session_id"]}: source video changed during extraction; finish copying first')
    arrays = {k: np.asarray(v, dtype=bool if k == 'face_found' else
              np.float64 if k == 'timestamps_ms' else np.float32) for k, v in values.items()}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp = cache_path.with_suffix('.tmp.npz')
    np.savez_compressed(temp, **arrays, identity=json.dumps(identity, sort_keys=True))
    temp.replace(cache_path)
    return arrays, identity


def rldd_windows(record, arrays, calibration, profile, profile_hash, calibration_s=60,
                 window_s=60, stride_s=10, timestamp_mode='pts'):
    """Use DROZY's metric engine and shared video definitions, same fixed schema."""
    timeline = make_timeline(arrays['landmarks'], arrays['timestamps_ms'], arrays['face_found'], calibration)
    metadata = {'status':'MINMAX_REFERENCES','method':'minmax_ear','profile_sha256':profile_hash}
    session = dict(record, time_source='video_pts_zero_origin' if timestamp_mode == 'pts'
                   else 'container_constant_fps_approximation')
    rows = window_metrics(timeline, session, None, metadata, window_s, stride_s)
    mar = arrays['mouth_aspect_ratio']
    mouth_valid = arrays['face_found'] & np.isfinite(mar) & (mar >= 0)
    add_video_indicators(rows, timeline, arrays, mar, mouth_valid, calibration_s)
    calibration_sid = profile['sources'][0]['session_id']
    for row in rows:
        start, end = row['window_start_s'], row['window_end_s']
        row.update({'participant_id':f'UTA_P{int(record["subject_id"]):03d}',
                    'session_order':None, 'baseline_kss':None, 'baseline_kss_definition':'NOT_AVAILABLE',
                    'kss_rating':None, 'perclos':row['perclos_combined'], 't_drive_min':end/60,
                    'calibration_session_id':calibration_sid,
                    'calibration_start_s':profile['calibration_interval_s'][0],
                    'calibration_end_s':profile['calibration_interval_s'][1],
                    'calibration_fit_scope':'first_available_subject_clip_prefix',
                    'calibration_overlap':record['session_id']==calibration_sid and start<profile['calibration_interval_s'][1],
                    'mouth_feature_source':'MediaPipe_inner_lip_landmarks',
                    'yawning_status':'HEURISTIC_MOUTH_OPENING_CANDIDATES_ONLY',
                    'head_nod_status':'UNVALIDATED_ABSOLUTE_PITCH_EXCURSIONS',
                    'label_rule':'alert(0)->0; drowsy(10)->1; low_vigilance(5)->blank (KSS 6-7 ambiguous at >=7)',
                    'dropped_frame_indicator':None,'dropped_frame_definition':'NOT_AVAILABLE_NO_CAPTURE_FRAME_GRID',
                    'psg_alignment_status':'NOT_AVAILABLE', 'pvt_available':False})
        # Availability flags are false; unavailable physiological measurements stay blank.
        for field in training_fieldnames():
            if field.startswith('psg_') and field.endswith('_available'):
                row[field] = False
        previous = next((r for r in rows if abs(r['window_end_s']-(end-300))<1e-6),None)
        row['delta_perclos_5min'] = row['perclos']-previous['perclos'] if previous else np.nan
    return rows


def _write_json(path, value):
    temp = Path(path).with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def export_rldd(dataset, output, model, subjects=None, sessions=None, max_seconds=None,
                calibration_s=60, window_s=60, stride_s=10, timestamp_mode='pts', inventory_only=False):
    for name, value in [('calibration_s',calibration_s),('window_s',window_s),('stride_s',stride_s),
                        ('max_seconds',max_seconds)]:
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f'{name} must be finite and positive')
    if max_seconds is not None and max_seconds < max(calibration_s,window_s):
        raise ValueError('Debug prefix must cover both calibration and a complete output window')
    records = discover_rldd(dataset)
    output, model = Path(output), Path(model)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output/'inventory.json', dataset_inventory(records))
    if inventory_only:
        return dataset_inventory(records)
    if not model.is_file():
        raise ValueError(f'Missing MediaPipe model: {model}')
    selected = records
    if subjects:
        wanted = {str(int(s)) for s in subjects}
        missing = wanted-{r['subject_id'] for r in records}
        if missing:
            raise ValueError(f'Unknown subjects: {sorted(missing)}')
        selected = [r for r in selected if r['subject_id'] in wanted]
    if sessions:
        missing = set(sessions)-{r['session_id'] for r in selected}
        if missing:
            raise ValueError(f'Unknown/excluded sessions: {sorted(missing)}')
        selected = [r for r in selected if r['session_id'] in sessions]
    if not selected:
        raise ValueError('No videos selected')
    # Pin enrollment independently of --sessions so rerunning subsets keeps endpoints stable.
    enrollment = {}
    for r in records:
        enrollment.setdefault(r['subject_id'],r)
    fields, profiles, rows, statuses = training_fieldnames(), {}, [], []
    audit = {'status':'RUNNING','dataset_id':'UTA_RLDD','selected_videos':len(selected),
             'partial_debug':max_seconds is not None,'options':{'calibration_s':calibration_s,
              'window_s':window_s,'stride_s':stride_s,'max_seconds':max_seconds,'timestamp_mode':timestamp_mode},
             'source_url':SOURCE_URL}
    started = time.monotonic()
    _write_json(output/'audit.json',audit)
    def save_progress():
        # Rebuild from the current selection; no stale session shards enter this CSV.
        temp = output/'uta_rldd_indicators.tmp.csv'
        write_csv(temp, rows, fields)
        temp.replace(output/'uta_rldd_indicators.csv')
        audit.update({'rows':len(rows),'columns':len(fields),'processed_videos':len(statuses),
                      'sessions':statuses,'elapsed_s':round(time.monotonic()-started,2),
                      'binary_label_counts':dict(Counter('blank' if r.get('drowsiness_label') is None
                                                        else str(r['drowsiness_label']) for r in rows)),
                      'missing_counts':{k:int(sum(r.get(k) is None or isinstance(r.get(k),(float,np.floating))
                                      and not np.isfinite(r[k]) for r in rows)) for k in fields}})
        _write_json(output/'audit.json',audit)
    for record in selected:
        sid, subject = record['session_id'],record['subject_id']
        try:
            if subject not in profiles:
                ref = enrollment[subject]
                ref_arrays, ref_identity = extract_rldd(ref,model,output,max_seconds,timestamp_mode)
                first = ref_arrays['timestamps_ms'][0]/1000
                profile = calibration_from_minmax(output/'cache',subject,ref['session_id'],first,first+calibration_s)
                (output/'calibration').mkdir(exist_ok=True)
                profile_path=output/'calibration'/f'subject_{subject}.json'
                _write_json(profile_path,profile)
                calibration = EyeCalibration(tuple(profile['ear_open']),tuple(profile['ear_closed']))
                profiles[subject] = (calibration,profile,hashlib.sha256(profile_path.read_bytes()).hexdigest())
                # Keep one enrollment extraction in memory to avoid loading it twice.
                enrollment_arrays = ref_arrays if sid == ref['session_id'] else None
            else:
                enrollment_arrays = None
            if enrollment_arrays is None:
                arrays, identity = extract_rldd(record,model,output,max_seconds,timestamp_mode)
            else:
                arrays, identity = enrollment_arrays, ref_identity
            calibration,profile,profile_hash = profiles[subject]
            new_rows=rldd_windows(record,arrays,calibration,profile,profile_hash,calibration_s,
                                  window_s,stride_s,timestamp_mode)
            if not new_rows:
                raise ValueError('No complete windows in this recording')
            rows.extend(new_rows)
            statuses.append({'session_id':sid,'status':'PARTIAL_DEBUG' if max_seconds is not None else 'COMPLETE',
                             'windows':len(new_rows),'frames':len(arrays['timestamps_ms']),
                             'class_id':record['class_id'],'fold':record['fold'],'extraction':identity})
            print(f'{sid}: {len(new_rows)} windows',flush=True)
        except (ValueError,OSError,RuntimeError) as error:
            statuses.append({'session_id':sid,'status':'FAILED','error':str(error)})
            print(f'{sid}: FAILED: {error}',flush=True)
        save_progress()
    failures = [s for s in statuses if s['status']=='FAILED']
    audit['status']='FAILED_SESSIONS' if failures else 'PARTIAL_DEBUG' if max_seconds is not None else 'COMPLETE_SELECTION'
    audit['calibration_overlap_windows']=int(sum(r['calibration_overlap'] for r in rows))
    save_progress()
    if failures:
        raise RuntimeError(f'{len(failures)} video(s) failed; see {output/"audit.json"}; successful rows saved')
    return audit
