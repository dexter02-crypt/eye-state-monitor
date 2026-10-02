# Design and source attribution

Pipeline: OpenCV BGR frame → MediaPipe Solutions Face Mesh → pixel-coordinate eye contours → coarse quality gates → per-eye EAR → calibrated hysteresis → timestamped state machine → display / optional local metrics.

For contour points p1…p6 around the eyelids:

`EAR = (distance(p2,p6) + distance(p3,p5)) / (2 × distance(p1,p4))`.

The EAR method is attributed to Tereza Soukupová and Jan Čech, *Real-Time Eye Blink Detection using Facial Landmarks*, CVWW 2016: https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf . This application does not reproduce their trained temporal classifier or inherit their benchmark results. The landmark detector, calibration procedure, and deterministic temporal logic here differ.

MediaPipe Face Mesh supplies pretrained facial landmarks locally. We request up to two faces so detected multi-person scenes can be refused. The project does not claim that landmark inference itself was trained or developed here. The geometric state machine supplies its own close/reopen hysteresis. Detection thresholds are not displayed as probabilities of correct eye state.

Coordinates are scaled by image width/height before distance calculation. Each eye's thresholds fall inside the gap between the closed-sample 90th percentile and open-sample 10th percentile; insufficient separation fails calibration. A small open-close gap can make calibration impossible under the current setup, which is preferable to asserting a misleading state.

State logic conservatively discards intervals whose endpoints disagree, are unknown, or are separated by more than 0.25 seconds. A valid CLOSED measurement immediately after a gap can restart closure timing at zero; unseen time is never counted. A blink requires an observed opening before the closure and an observed reopening within its duration band. Long episodes count once and are not also counted as blinks.

These checks improve the decision procedure's inspectability. They do not establish a measured accuracy improvement over another project; that requires a held-out comparative experiment.
