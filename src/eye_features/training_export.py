"""Reproducible multimodal DROZY feature export; no model fitting or imputation."""

from datetime import datetime, timedelta
from pathlib import Path
import csv
import hashlib
import json

import numpy as np

from .calibration import EyeCalibration
from .drozy import drozy_records
from .dataset_manifest import load_pvt
from .offline import (calibration_from_minmax, make_timeline, window_metrics,
                      write_csv, EYE_NAMES)

SYNTHETIC_FIELDS = [
    'participant_id', 'session_order', 'start_time', 'sleep_duration', 'wake_time',
    'elapsed_time_awake', 'baseline_kss', 'kss_rating', 'eye_openness', 'perclos',
    'blink_rate_per_min', 'avg_blink_duration_ms', 'blink_duration_variability_ms',
    'yawning_rate_per_min', 'head_pitch_deg', 'head_yaw_deg', 'head_roll_deg',
    'observer_rating', 'camera_frame_timestamp', 'dropped_frame_indicator',
    'video_quality_score', 'drowsiness_label',
]
BANDS = {'delta': (.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}


def read_edf(path):
    """Read continuous EDF int16 signals, preserving physical calibration/units.

    Specification: https://www.edfplus.info/specs/edf.html
    Reject EDF+D and unsupported headers instead of silently misaligning data.
    """
    with Path(path).open('rb') as f:
        h = f.read(256)
        if len(h) != 256 or h[:8].strip() != b'0' or b'EDF+D' in h[192:236]:
            raise ValueError('Expected continuous EDF version 0')
        n, records, duration = int(h[252:256]), int(h[236:244]), float(h[244:252])
        if n <= 0 or records <= 0 or duration <= 0 or int(h[184:192]) != 256 * (n + 1):
            raise ValueError('Invalid or unsupported EDF header')
        fields = {}
        for name, width in [('label',16), ('transducer',80), ('unit',8), ('pmin',8),
                            ('pmax',8), ('dmin',8), ('dmax',8), ('filter',80),
                            ('samples',8), ('reserved',32)]:
            raw = f.read(width*n)
            fields[name] = [raw[i*width:(i+1)*width].decode('ascii').strip() for i in range(n)]
        samples = np.array(fields['samples'], dtype=int)
        raw = np.frombuffer(f.read(), dtype='<i2')
    if np.any(samples <= 0) or raw.size != records * samples.sum():
        raise ValueError('EDF payload length does not match header')
    raw = raw.reshape(records, samples.sum())
    result, offset = {}, 0
    for i, label in enumerate(fields['label']):
        lo, hi = float(fields['dmin'][i]), float(fields['dmax'][i])
        pmin, pmax = float(fields['pmin'][i]), float(fields['pmax'][i])
        if hi <= lo or label in result:
            raise ValueError('Invalid EDF channel calibration or duplicate label')
        digital = raw[:, offset:offset+samples[i]].reshape(-1)
        result[label] = {'values': (digital.astype(float)-lo)*(pmax-pmin)/(hi-lo)+pmin,
                         'fs': samples[i]/duration, 'unit': fields['unit'][i],
                         'clipped': (digital <= lo) | (digital >= hi)}
        offset += samples[i]
    return datetime.strptime(h[168:184].decode(), '%d.%m.%y%H.%M.%S'), result


def bandpowers(values, fs):
    """Hann-window Welch PSD, 4-second segments, 50% overlap, mean removed."""
    n = round(4*fs)
    if len(values) < n or not np.all(np.isfinite(values)):
        return {}
    chunks = np.lib.stride_tricks.sliding_window_view(values, n)[::n//2]
    taper = np.hanning(n)
    fft = np.fft.rfft((chunks-chunks.mean(axis=1, keepdims=True))*taper, axis=1)
    psd = np.mean(abs(fft)**2/(fs*np.sum(taper*taper)), axis=0)
    psd[1:-1] *= 2
    freq = np.fft.rfftfreq(n, 1/fs)
    df = fs/n
    powers = {name: float(psd[(freq >= a) & (freq < b)].sum()*df)
              for name, (a,b) in BANDS.items()}
    total = sum(powers.values())
    out = {f'{name}_power': v for name,v in powers.items()}
    out.update({f'{name}_relative': v/total if total > 0 else np.nan for name,v in powers.items()})
    out['theta_alpha_ratio'] = powers['theta']/powers['alpha'] if powers['alpha'] > 0 else np.nan
    out['theta_plus_alpha_beta_ratio'] = (powers['theta']+powers['alpha'])/powers['beta'] if powers['beta'] > 0 else np.nan
    return out


def completed_runs(mask, valid, times, min_s, max_s=np.inf):
    """Complete threshold runs only; do not bridge missing frames or large gaps."""
    period = float(np.median(np.diff(times)))
    gap = np.diff(times) > 2.5*period
    runs, start = [], None
    for i in range(len(times)):
        broken = i > 0 and gap[i-1]
        if broken or not valid[i]:
            start = None
        if valid[i] and mask[i]:
            if start is None and i > 0 and valid[i-1] and not mask[i-1] and not broken:
                start = i
        elif start is not None:
            duration = times[i]-times[start]
            if min_s <= duration <= max_s:
                runs.append((float(times[start]), float(times[i]), float(duration)))
            start = None
    return runs


def psg_features(signals, offset, start, end):
    out = {'psg_alignment_status': 'EDF_HEADER_SECONDS_APPROXIMATE',
           'psg_alignment_uncertainty_s': 1.0, 'psg_window_complete': True}
    for label in ('Fz','Cz','C3','C4','Pz','Oz','EOG-V','EOG-H','EMG','ECG'):
        prefix = 'psg_'+label.lower().replace('-', '_')
        signal = signals.get(label)
        out[prefix+'_available'] = signal is not None
        if signal is None:
            continue
        fs = signal['fs']
        a, b = round((start+offset)*fs), round((end+offset)*fs)
        out[prefix+'_unit'] = signal['unit']
        if a < 0 or b > len(signal['values']) or b <= a:
            out['psg_window_complete'] = False
            continue
        x = signal['values'][a:b]
        out.update({prefix+'_mean': float(x.mean()), prefix+'_std': float(x.std()),
                    prefix+'_rms': float(np.sqrt(np.mean(x*x))),
                    prefix+'_peak_to_peak': float(np.ptp(x)),
                    prefix+'_clipped_fraction': float(signal['clipped'][a:b].mean())})
        if label in ('Fz','Cz','C3','C4','Pz','Oz'):
            out.update({prefix+'_'+k: v for k,v in bandpowers(x,fs).items()})
    return out


def training_fieldnames():
    """Stable 273-column contract shared by DROZY and UTA-RLDD."""
    return json.loads(Path(__file__).with_name('training_schema.json').read_text())


def add_video_indicators(rows, timeline, arrays, mar, mouth_valid, calibration_s=60.0):
    """Add the same blink, mouth and head indicators for both datasets."""
    times = timeline.timestamps_s
    mouth_events = completed_runs(mar >= .6, mouth_valid, times, 1.0)
    pitch = arrays['head_pose'][:, 0]
    head_valid = arrays['face_found'] & np.isfinite(pitch)
    base_mask = head_valid & (times <= times[0] + calibration_s)
    pitch_baseline = float(np.median(pitch[base_mask])) if base_mask.any() else np.nan
    nod_events = completed_runs(abs(pitch-pitch_baseline) >= 10, head_valid, times, .3, 4.0)
    for row in rows:
        start, end = row['window_start_s'], row['window_end_s']
        mask = (times >= start) & (times < end)
        for eye in (*EYE_NAMES,'both'):
            blinks = [e['duration_s']*1000 for e in timeline.events if e['eye']==eye
                      and e['candidate_blink'] and start<=e['observed_end_s']<end]
            closures = [e for e in timeline.events if e['eye']==eye and e['prolonged_closure']
                        and start<=e['observed_end_s']<end]
            row[f'blink_rate_{eye}_per_min'] = len(blinks)*60/(end-start)
            row[f'blink_duration_std_{eye}_ms'] = float(np.std(blinks)) if len(blinks)>=2 else np.nan
            row[f'slow_blink_rate_{eye}_per_min'] = sum(x>=300 for x in blinks)*60/(end-start)
            row[f'longest_completed_closure_{eye}_ms'] = max((e['duration_s']*1000 for e in closures),default=np.nan)
        row['blink_rate_per_min'] = row['blink_rate_both_per_min']
        row['avg_blink_duration_ms'] = row['mean_blink_duration_both_ms']
        row['blink_duration_variability_ms'] = row['blink_duration_std_both_ms']
        for axis,i in [('pitch',0),('yaw',1),('roll',2)]:
            vals = arrays['head_pose'][mask & arrays['face_found'],i]
            vals = vals[np.isfinite(vals)]
            row[f'head_{axis}_deg'] = float(vals.mean()) if len(vals) else np.nan
            row[f'head_{axis}_std_deg'] = float(vals.std()) if len(vals) else np.nan
            row[f'head_{axis}_range_deg'] = float(np.ptp(vals)) if len(vals) else np.nan
        vals=mar[mask&mouth_valid]
        row.update({'mouth_aspect_ratio_mean':float(vals.mean()) if len(vals) else np.nan,
                    'mouth_aspect_ratio_max':float(vals.max()) if len(vals) else np.nan,
                    'mouth_valid_fraction':float(mouth_valid[mask].mean()),
                    'mouth_open_candidate_rate_per_min':sum(start<=b<end for a,b,d in mouth_events)*60/(end-start),
                    'head_nod_candidate_rate_per_min':sum(start<=b<end for a,b,d in nod_events)*60/(end-start)})
        # Compatibility alias, explicitly identified as a heuristic candidate rate.
        row['yawning_rate_per_min'] = row['mouth_open_candidate_rate_per_min']


def export_training(dataset, processed, output, calibration_s=60.0, kss_threshold=7):
    dataset, processed, output = Path(dataset), Path(processed), Path(output)
    if not np.isfinite(calibration_s) or calibration_s <= 0:
        raise ValueError('Calibration duration must be positive')
    if not 2 <= kss_threshold <= 9:
        raise ValueError('KSS threshold must be 2..9')
    records = [r for r in drozy_records(dataset) if r['video_available']]
    records.sort(key=lambda r: (int(r['subject_id']), int(r['test_id'])))
    output.mkdir(parents=True, exist_ok=True)
    profiles, all_rows, inventory = {}, [], []
    baseline = {str(r['subject_id']): next((x['kss'] for x in records
                 if str(x['subject_id']) == str(r['subject_id'])), None) for r in records}
    for record in records:
        sid, subject = record['session_id'], str(record['subject_id'])
        cache_path = processed/'cache'/f'{sid}.npz'
        with np.load(cache_path, allow_pickle=False) as cache:
            arrays = {k: cache[k] for k in ('landmarks','timestamps_ms','face_found','head_pose')}
            identity = json.loads(str(cache['identity']))
        if str(identity['subject_id']) != subject or identity['dataset_id'] != 'DROZY':
            raise ValueError(f'{sid}: cache identity mismatch')
        source_ts = np.loadtxt(dataset/'timestamps'/f'{sid}.txt', ndmin=2)[:,7]
        if not np.array_equal(source_ts, arrays['timestamps_ms']):
            raise ValueError(f'{sid}: cache/source timestamps differ')
        if subject not in profiles:
            start = arrays['timestamps_ms'][0]/1000
            profile = calibration_from_minmax(processed/'cache', subject, sid, start, start+calibration_s)
            profiles[subject] = profile
            directory = output/'calibration'
            directory.mkdir(exist_ok=True)
            (directory/f'subject_{subject}.json').write_text(json.dumps(profile,indent=2)+'\n')
        profile = profiles[subject]
        calibration = EyeCalibration(tuple(profile['ear_open']),tuple(profile['ear_closed']))
        metadata = {'status':'MINMAX_REFERENCES','method':'minmax_ear',
                    'profile_sha256': hashlib.sha256(json.dumps(profile,sort_keys=True).encode()).hexdigest()}
        timeline = make_timeline(arrays['landmarks'], arrays['timestamps_ms'], arrays['face_found'],calibration)
        times = timeline.timestamps_s
        pvt = load_pvt(record)
        rows = window_metrics(timeline,record,pvt,metadata)
        native = np.loadtxt(dataset/'annotations-auto'/f'{sid}-s2.txt').reshape(-1,68,2)
        if len(native) != len(times):
            raise ValueError(f'{sid}: native landmarks are not frame-aligned')
        width = np.linalg.norm(native[:,60]-native[:,64],axis=1)
        mar = np.divide(sum(np.linalg.norm(native[:,a]-native[:,b],axis=1)
                            for a,b in ((61,67),(62,66),(63,65))),3*width,
                        out=np.full(len(width),np.nan),where=width>0)
        mouth_valid = np.isfinite(native).all(axis=(1,2)) & np.isfinite(mar)
        add_video_indicators(rows, timeline, arrays, mar, mouth_valid, calibration_s)
        interpolation = np.loadtxt(dataset/'interpIndices'/f'{sid}.txt',ndmin=2)
        if interpolation.shape != (18000,3):
            raise ValueError(f'{sid}: unexpected interpolation grid')
        edf_start, signals = read_edf(dataset/'psg'/f'{sid}.edf')
        pvt_origin = datetime.strptime((dataset/'pvt-rt'/f'{sid}.csv').read_text().splitlines()[0],'%Y-%m-%d_%H.%M.%S.%f')
        offset = (pvt_origin-edf_start).total_seconds()
        if abs(offset) > 2:
            raise ValueError(f'{sid}: unexplained EDF/PVT header offset {offset}')
        for row in rows:
            start,end = row['window_start_s'],row['window_end_s']
            mask = (times>=start)&(times<end)
            row.update({'participant_id':f'P{int(subject):03d}', 'session_order':int(record['test_id']),
                        'start_time':pvt_origin.isoformat(), 'baseline_kss':baseline[subject],
                        'baseline_kss_definition':'first_available_session_score',
                        'kss_rating':record['kss'], 'drowsiness_label':int(record['kss']>=kss_threshold),
                        'label_scope':'session','label_source':'DROZY/KSS.txt',
                        'label_rule':f'kss >= {kss_threshold}; analysis convention',
                        'camera_frame_timestamp':(pvt_origin+timedelta(seconds=start)).isoformat(),
                        'perclos':row['perclos_combined'], 't_drive_min':end/60,
                        'calibration_session_id':profile['sources'][0]['session_id'],
                        'calibration_start_s':profile['calibration_interval_s'][0],
                        'calibration_end_s':profile['calibration_interval_s'][1],
                        'calibration_fit_scope':'first_available_subject_session_prefix',
                        'calibration_overlap': sid==profile['sources'][0]['session_id'] and start<profile['calibration_interval_s'][1],
                        'mouth_feature_source':'DROZY_native_auto_68_landmarks',
                        'yawning_status':'HEURISTIC_MOUTH_OPENING_CANDIDATES_ONLY',
                        'head_nod_status':'UNVALIDATED_ABSOLUTE_PITCH_EXCURSIONS',
                        'sleep_duration':None,'wake_time':None,'elapsed_time_awake':None,
                        'observer_rating':None,'video_quality_score':None})
            reactions = np.array([e['reaction_time_ms'] for e in pvt if start<=e['stimulus_s']<end and e['response_s']<end])
            row['pvt_rt_std_ms'] = float(reactions.std()) if len(reactions)>=2 else np.nan
            row['pvt_rt_p90_ms'] = float(np.quantile(reactions,.9)) if len(reactions) else np.nan
            row['pvt_lapse_fraction'] = float(np.mean(reactions>=500)) if len(reactions) else np.nan
            row['pvt_false_start_count'] = int(np.sum(reactions<100))
            grid = interpolation[round(start*30):round(end*30)]
            missing = grid[:,2] != 1.0
            row['interpolated_grid_fraction'] = float(missing.mean())
            row['dropped_frame_indicator'] = int(missing.any())
            row['dropped_frame_definition'] = 'any_interpolated_sample_on_DROZY_30Hz_grid'
            row.update(psg_features(signals,offset,start,end))
        for i,row in enumerate(rows):
            previous = next((r for r in rows[:i] if abs(r['window_end_s']-(row['window_end_s']-300))<1e-6),None)
            row['delta_perclos_5min'] = row['perclos']-previous['perclos'] if previous else np.nan
        all_rows.extend(rows)
        inventory.append({'session_id':sid,'kss':record['kss'],'windows':len(rows),
                          'edf_pvt_header_offset_s':offset,'cache_sha256':hashlib.sha256(cache_path.read_bytes()).hexdigest()})
        print(f'{sid}: {len(rows)} windows, KSS {record["kss"]}',flush=True)
    fields = training_fieldnames()
    write_csv(output/'drozy_indicators.csv',all_rows,fields)
    audit = {'rows':len(all_rows),'columns':len(fields),'participants':len(profiles),
             'sessions':inventory,'kss_threshold':kss_threshold,'calibration_duration_s':calibration_s,
             'missing_counts':{k:int(sum(r.get(k) is None or isinstance(r.get(k),(float,np.floating))
                                and not np.isfinite(r[k]) for r in all_rows)) for k in fields},
             'binary_label_counts':{str(v):sum(r['drowsiness_label']==v for r in all_rows) for v in (0,1)},
             'calibration_overlap_windows':int(sum(r['calibration_overlap'] for r in all_rows))}
    (output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    return audit
