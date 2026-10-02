#!/usr/bin/env python3
"""Eye state and prolonged closure experiment. Never a sleep/driver-safety diagnosis."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import io
import json
import math
from pathlib import Path
import sys
import time
from local_support import ROOT, capture, overlay, read_json, unique_output, write_json, write_new
from eye_state.core import EyeTracker, Thresholds, calibrate_eye, evaluate


def load_profile(path):
    data = read_json(Path(path))
    if not isinstance(data, dict) or data.get('schema') != 1 or data.get('method') != 'per-eye-EAR-calibration':
        raise ValueError('Unsupported calibration profile.')
    return Thresholds(**data['left']), Thresholds(**data['right'])


def demo(output=None):
    tracker = EyeTracker(Thresholds(.16,.22), Thresholds(.16,.22))
    rows = []
    for i in range(181):
        t = i/30
        closed = .9 <= t < 1.1 or 2 <= t < 4
        missing = 4.5 <= t < 5
        ear = .1 if closed else .3
        rows.append(asdict(tracker.update(t, None if missing else ear, None if missing else ear)))
    result = {'kind': 'SYNTHETIC_LOGIC_DEMO_NOT_CAMERA_ACCURACY', 'observations': rows,
              'blinks': rows[-1]['blinks'], 'prolonged_events': rows[-1]['prolonged_events']}
    path = Path(output) if output else unique_output('.json')
    write_json(path, result)
    print(f'Synthetic geometry trace: {result["blinks"]} blink, {result["prolonged_events"]} prolonged event.\n{path}')


def run_camera(args, calibrating=False):
    import cv2
    from eye_state.vision import create_detector, infer, measure_frame
    profile_path = Path(args.profile)
    if calibrating and (profile_path.exists() or profile_path.is_symlink()):
        raise ValueError('Profile exists. Keep it or select a NEW --profile path; no automatic overwrite.')
    tracker = None if calibrating else EyeTracker(*load_profile(profile_path), prolonged_s=args.prolonged)
    samples = {'open': [], 'closed': []}
    phase, phase_start = None, None
    last_ms = -1
    start = time.monotonic()
    rows = []
    print('Use a stationary camera and ONE consenting person. Q/Esc exits. No frames are saved.')
    if calibrating:
        print('Press O and keep eyes naturally open for 3 seconds. Press C, close naturally for 5 seconds, then reopen.')
    previous_center = None
    try:
        with create_detector(video=True) as detector, capture(args.camera, getattr(args,'video',None)) as cap:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if getattr(args,'video',None) and (not math.isfinite(fps) or fps <= 0):
                raise ValueError('Local video needs valid constant-frame-rate metadata.')
            index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    if getattr(args,'video',None): break
                    raise ValueError('Camera stopped returning frames. Session stopped; no awake inference.')
                t = index/fps if getattr(args,'video',None) else time.monotonic()-start
                timestamp = max(last_ms+1, round(t*1000))
                result = infer(detector, frame, timestamp)
                last_ms = timestamp
                left,right,reason,contours = measure_frame(result, frame)
                if not reason:
                    center = tuple(sum(p[k] for c in contours for p in c)/12 for k in (0,1))
                    if previous_center and math.dist(previous_center,center) > frame.shape[1]*.12:
                        left=right=None;reason='POSITION_JUMP'
                    previous_center=center
                else:
                    previous_center=None
                display = frame.copy()
                for contour in contours:
                    for x,y in contour:
                        cv2.circle(display,(int(x),int(y)),2,(80,220,120),-1)
                if calibrating:
                    status = reason or 'Ready'
                    if phase is not None:
                        elapsed = time.monotonic()-phase_start
                        delay = 2 if phase=='closed' else 0
                        if elapsed >= delay and not reason:
                            samples[phase].append((left,right))
                        status = f'{phase.upper()} calibration: {max(0,delay+3-elapsed):.1f}s remaining'
                        if elapsed >= delay+3:
                            phase=None
                            if samples['open'] and samples['closed']:
                                limits=[]
                                for j in (0,1):
                                    limits.append(calibrate_eye([v[j] for v in samples['open']], [v[j] for v in samples['closed']]))
                                write_json(profile_path, {'schema':1,'method':'per-eye-EAR-calibration',
                                    'left':asdict(limits[0]),'right':asdict(limits[1]),
                                    'samples':{k:len(v) for k,v in samples.items()},
                                    'note':'Local personal thresholds, not an accuracy evaluation.'})
                                print(f'Calibration saved: {profile_path}')
                                return
                    lines = ['EYE STATE | CALIBRATION', status,
                             f'Open samples: {len(samples["open"])} | Closed samples: {len(samples["closed"])}',
                             'O: open 3s | C: close naturally 5s, then reopen | Q: exit']
                else:
                    observation = tracker.update(t,left,right,reason)
                    row=asdict(observation)
                    if args.csv:
                        if len(rows)>=200_000: raise ValueError('CSV frame limit reached. Use a shorter session.')
                        rows.append(row)
                    state = 'PROLONGED EYE CLOSURE' if observation.prolonged else observation.state
                    ratio = observation.closed_fraction_of_known
                    lines=['EYE STATE | not a sleep, medical, or driving-safety assessment',
                           f'{state} | {reason}',
                           f'Closed: {observation.closed_for_s:.2f}s | Blinks: {observation.blinks} | Closure events: {observation.prolonged_events}',
                           f'30s known-time coverage: {observation.known_fraction:.0%} | Closed fraction of known: {ratio:.1%}' if ratio is not None else 'Insufficient observed time',
                           'Q/Esc: exit | No frames recorded; metrics saved only with --csv']
                if not getattr(args,'no_display',False):
                    cv2.imshow('Eye State Monitor',overlay(display,lines))
                    key=cv2.waitKey(1)&0xFF
                    if key in (ord('q'),27):break
                    if calibrating and key in (ord('o'),ord('c')):
                        phase='open' if key==ord('o') else 'closed'
                        samples[phase]=[];phase_start=time.monotonic()
                index+=1
                if getattr(args,'max_frames',None) and index>=args.max_frames:break
    finally:
        cv2.destroyAllWindows()
        if rows and getattr(args,'csv',None):
            stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            write_new(Path(args.csv),stream.getvalue().encode())
            print(f'Local metrics saved (no video): {args.csv}')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('check',help='Initialize Face Mesh and run blank-image inference; not an accuracy test')
    p=sub.add_parser('demo');p.add_argument('--output')
    p=sub.add_parser('evaluate');p.add_argument('csv');p.add_argument('--output')
    for name in ('calibrate','camera','video'):
        p=sub.add_parser(name);p.add_argument('--camera',type=int,default=0)
        p.add_argument('--profile',default=str(ROOT/'.local/calibration.json'))
        if name!='calibrate':
            p.add_argument('--prolonged',type=float,default=1.5)
            p.add_argument('--csv',help='Explicit local numeric metrics export; never uploads')
            p.add_argument('--max-frames',type=int)
        if name=='video':
            p.add_argument('video');p.add_argument('--no-display',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.command=='check':
            import numpy as np
            from eye_state.vision import create_detector,infer
            with create_detector() as detector:
                result=infer(detector,np.zeros((480,640,3),dtype=np.uint8))
            if result.face_landmarks:raise ValueError('Unexpected face on blank smoke-test image.')
            print('MediaPipe Face Mesh initialized; blank-image inference returned zero faces. Camera accuracy is unmeasured.')
        elif args.command=='demo':demo(args.output)
        elif args.command=='evaluate':
            path=Path(args.csv)
            if not path.is_file() or path.stat().st_size>10_000_000:raise ValueError('Evaluation CSV missing or too large.')
            with path.open(newline='',encoding='utf-8-sig') as stream: result=evaluate(csv.DictReader(stream))
            if args.output:write_json(Path(args.output),result)
            print(json.dumps(result,indent=2,allow_nan=False))
        else:
            if getattr(args,'max_frames',None) is not None and args.max_frames<1:raise ValueError('--max-frames must be positive.')
            if getattr(args,'csv',None) and (Path(args.csv).exists() or Path(args.csv).is_symlink()):raise ValueError('CSV destination already exists.')
            run_camera(args,args.command=='calibrate')
        return 0
    except (OSError,ValueError,RuntimeError,KeyError,TypeError,ImportError) as exc:
        print(f'Error: {exc}',file=sys.stderr);return 2
    except KeyboardInterrupt:return 130
if __name__=='__main__':raise SystemExit(main())
