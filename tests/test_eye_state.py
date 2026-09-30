import json,math,subprocess,sys,tempfile,unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace as NS
from eye_state.core import Thresholds,EyeTracker,eye_aspect_ratio,calibrate_eye,classify,evaluate
from eye_state.vision import measure,LEFT,RIGHT
from local_support import write_json,write_new,read_json

class GeometryTests(unittest.TestCase):
    def setUp(self):self.points=[(0,0),(2,1.5),(8,1.5),(10,0),(8,-1.5),(2,-1.5)]
    def test_ear_known_value(self):self.assertAlmostEqual(eye_aspect_ratio(self.points),.3)
    def test_ear_scale_translation_invariance(self):self.assertAlmostEqual(eye_aspect_ratio([(x*3+17,y*3-9) for x,y in self.points]),.3)
    def test_ear_in_plane_rotation(self):self.assertAlmostEqual(eye_aspect_ratio([(-y,x) for x,y in self.points]),.3)
    def test_degenerate_width_rejected(self):
        self.points[3]=self.points[0]
        with self.assertRaises(ValueError):eye_aspect_ratio(self.points)
    def test_nonfinite_rejected(self):
        self.points[2]=(float('nan'),0)
        with self.assertRaises(ValueError):eye_aspect_ratio(self.points)
    def test_threshold_order(self):
        for values in [(0.2,0.1),(.2,.2),(-.1,.3),(0,float('inf'))]:
            with self.subTest(values=values),self.assertRaises(ValueError):Thresholds(*values)
    def test_calibration_separates_states(self):
        limits=calibrate_eye([.30,.32]*15,[.05,.08]*15)
        self.assertLess(.08,limits.close);self.assertLess(limits.close,limits.reopen);self.assertLess(limits.reopen,.30)
    def test_bad_calibration_refused(self):
        for a,b in [([.3]*19,[.1]*20),([.2]*25,[.21]*25),([.2]*25,[.19]*25),([float('nan')]*25,[.1]*25)]:
            with self.subTest(a=a[:1],b=b[:1]),self.assertRaises(ValueError):calibrate_eye(a,b)
    def test_hysteresis_boundaries(self):
        t=Thresholds(.16,.22)
        self.assertEqual(classify(.16,'OPEN',t),'CLOSED');self.assertEqual(classify(.22,'CLOSED',t),'OPEN')
        self.assertEqual(classify(.19,'CLOSED',t),'CLOSED');self.assertEqual(classify(.19,'OPEN',t),'OPEN');self.assertEqual(classify(.19,'UNKNOWN',t),'UNKNOWN')
    def test_no_face_is_unknown_measurement(self):self.assertEqual(measure([],640,480)[2],'NO_FACE')
    def test_multiple_faces_refused(self):self.assertEqual(measure([[],[]],640,480)[2],'MULTIPLE_FACES')
    def face(self):
        p=[NS(x=.5,y=.5) for _ in range(478)]
        for indices,cx in ((LEFT,.65),(RIGHT,.35)):
            for index,(x,y) in zip(indices,[(-.05,0),(-.03,-.015),(.03,-.015),(.05,0),(.03,.015),(-.03,.015)]):p[index]=NS(x=cx+x,y=.4+y)
        p[1]=NS(x=.5,y=.55)
        return p
    def test_geometry_adapter_uses_pixel_aspect_ratio(self):
        a,b,reason,_=measure([self.face()],1000,500)
        self.assertEqual(reason,'');self.assertAlmostEqual(a,.15);self.assertAlmostEqual(b,.15)
    def test_geometry_adapter_rejects_small_eyes(self):self.assertEqual(measure([self.face()],120,100)[2],'TOO_FAR_AWAY')
    def test_invalid_landmarks_refused(self):
        face=self.face();face[LEFT[2]]=NS(x=float('nan'),y=.4)
        self.assertEqual(measure([face],640,480)[2],'INVALID_LANDMARKS')
    def test_out_of_frame_refused(self):
        face=self.face();face[LEFT[0]]=NS(x=1.1,y=.4)
        self.assertEqual(measure([face],640,480)[2],'EYES_OUT_OF_FRAME')

class TemporalTests(unittest.TestCase):
    def tracker(self,**kwargs):return EyeTracker(Thresholds(.16,.22),Thresholds(.16,.22),**kwargs)
    def test_normal_blink_counted_once(self):
        tr=self.tracker()
        for t,ear in [(0,.3),(.1,.1),(.2,.1),(.3,.3),(.4,.3)]:r=tr.update(t,ear,ear)
        self.assertEqual(r.blinks,1)
    def test_single_frame_noise_not_blink(self):
        tr=self.tracker();tr.update(0,.3,.3);tr.update(.02,.1,.1);r=tr.update(.04,.3,.3)
        self.assertEqual(r.blinks,0)
    def test_prolonged_event_once_per_episode(self):
        tr=self.tracker();tr.update(0,.3,.3)
        for i in range(1,31):r=tr.update(i/10,.1,.1)
        self.assertTrue(r.prolonged);self.assertEqual(r.prolonged_events,1)
        r=tr.update(3.1,.3,.3);self.assertEqual(r.blinks,0);self.assertFalse(r.prolonged)
    def test_initial_closed_not_invented_blink(self):
        tr=self.tracker();tr.update(0,.1,.1);tr.update(.1,.1,.1);r=tr.update(.2,.3,.3);self.assertEqual(r.blinks,0)
    def test_absence_breaks_continuity(self):
        tr=self.tracker();tr.update(0,.3,.3);tr.update(.1,.1,.1);r=tr.update(.2,None,None)
        self.assertEqual(r.state,'UNKNOWN');self.assertEqual(r.closed_for_s,0)
        tr.update(.3,.1,.1);r=tr.update(.4,.3,.3);self.assertEqual(r.blinks,0)
    def test_asymmetric_eye_not_both_closed(self):
        tr=self.tracker();tr.update(0,.3,.3);r=tr.update(.1,.1,.3)
        self.assertEqual(r.state,'ASYMMETRIC');self.assertFalse(r.prolonged)
    def test_large_gap_resets_timer(self):
        tr=self.tracker();tr.update(0,.3,.3);tr.update(.1,.1,.1);r=tr.update(5,.1,.1)
        self.assertEqual(r.closed_for_s,0);self.assertFalse(r.prolonged);self.assertEqual(r.reason,'SAMPLING_GAP_RESET')
    def test_time_reversal_and_equal_rejected(self):
        tr=self.tracker();tr.update(1,.3,.3)
        for t in [1,.5,-1,float('nan')]:
            with self.subTest(t=t),self.assertRaises(ValueError):tr.update(t,.3,.3)
    def test_invalid_measurement_not_awake(self):
        tr=self.tracker();r=tr.update(0,float('nan'),.3);self.assertEqual(r.state,'UNKNOWN')
    def test_time_weighted_coverage(self):
        tr=self.tracker();tr.update(0,.3,.3);tr.update(.2,.3,.3);tr.update(.3,None,None);r=tr.update(.4,None,None)
        self.assertAlmostEqual(r.known_time_s,.2);self.assertAlmostEqual(r.known_fraction,.5)
    def test_zero_coverage_has_no_closure_fraction(self):
        tr=self.tracker();r=tr.update(0,None,None);self.assertIsNone(r.closed_fraction_of_known)
    def test_window_expires_old_intervals(self):
        tr=self.tracker(window_s=1)
        for i in range(21):r=tr.update(i/10,.3,.3)
        self.assertAlmostEqual(r.known_time_s,1);self.assertAlmostEqual(r.known_fraction,1)
    def test_configuration_rejects_invalid_durations(self):
        for kw in [{'prolonged_s':.3},{'min_blink_s':0},{'max_gap_s':-1},{'window_s':float('inf')}]:
            with self.subTest(kw=kw),self.assertRaises(ValueError):self.tracker(**kw)

class EvaluationAndIOTests(unittest.TestCase):
    def test_metrics_unknowns_not_hidden(self):
        r=evaluate([{'truth':t,'prediction':p} for t,p in [('CLOSED','CLOSED'),('OPEN','CLOSED'),('CLOSED','UNKNOWN'),('OPEN','OPEN')]])
        self.assertEqual(r['closed_precision'],.5);self.assertEqual(r['closed_recall_including_abstentions'],.5)
        self.assertEqual(r['coverage'],.75);self.assertEqual(r['overall_correct_fraction'],.5)
    def test_no_predictions_does_not_invent_precision(self):self.assertIsNone(evaluate([{'truth':'CLOSED','prediction':'UNKNOWN'}])['closed_precision'])
    def test_invalid_labels_refused(self):
        with self.assertRaises(ValueError):evaluate([{'truth':'SLEEPING','prediction':'CLOSED'}])
    def test_exclusive_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder).resolve()/'a.json';write_json(p,{'a':1})
            with self.assertRaises(FileExistsError):write_json(p,{'a':2})
            self.assertEqual(read_json(p),{'a':1})
    def test_symlink_output_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();(root/'original').write_text('keep');(root/'link').symlink_to(root/'original')
            with self.assertRaises(ValueError):write_new(root/'link',b'replace')
    def test_cli_synthetic_demo(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder).resolve()/'demo.json';r=subprocess.run([sys.executable,'app.py','demo','--output',str(out)],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr);data=read_json(out)
            self.assertEqual(data['blinks'],1);self.assertEqual(data['prolonged_events'],1)
    def test_missing_profile_fails_without_camera(self):
        r=subprocess.run([sys.executable,'app.py','camera','--profile','missing-profile-file.json'],capture_output=True,text=True)
        self.assertEqual(r.returncode,2);self.assertIn('Missing',r.stderr)
