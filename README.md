# Eye State Monitor

A small, inspectable webcam experiment for **OPEN / CLOSED / ASYMMETRIC / UNKNOWN** eye state, blink events, and prolonged eye closure. Uses pretrained MediaPipe face landmarks, personal eye calibration, and time-based event logic—not an eye model trained from scratch.

**It does not determine whether a person is asleep or awake. Never use it as a driving alarm, medical device, attendance monitor, or proof of attention. No accuracy percentage has been measured.** The shipped evidence covers synthetic geometry/temporal tests, not human-video inference. Native model startup and camera operation must be checked locally.

## Start from this repository

```bash
PYTHON=/opt/homebrew/bin/python3.12 bash setup.sh
.venv/bin/python app.py check
.venv/bin/python app.py calibrate
.venv/bin/python app.py camera
```

Run commands separately; stop when any returns an error. `check` initializes the local MediaPipe Face Mesh backend and processes a blank image; success proves native backend startup, not detection accuracy.

During calibration, keep ONE face centred, both eyes clearly visible, and the camera stationary. Use ordinary frontal lighting. Press **O**, keep your eyes naturally open for three seconds. Then press **C**, close naturally for five seconds and reopen. The first two seconds of the closed phase are a transition delay; the next three collect samples. Do not squeeze the eyes shut. Calibration requires at least 20 usable samples in each phase and refuses overlapping open/closed distributions.

The profile is saved to `.local/calibration.json`. Profiles are session/person/camera dependent. It will not overwrite a previous profile. To recalibrate without deleting anything:

```bash
.venv/bin/python app.py calibrate --profile .local/session-2.json
.venv/bin/python app.py camera --profile .local/session-2.json
```

Press Q or Esc in the video window to stop. `--camera 1` selects another local camera. The program only acquires the camera after `camera` or `calibrate`, never during setup, `check`, or ordinary unit-test setup.

## What is different from a one-threshold demo?

Each eye is calibrated separately. Different close/reopen thresholds reduce state flicker. A short open→closed→open episode can increment the blink count; a continuously observed longer closure increments a separate event count once. Multiple faces, missing landmarks, coarse pose/lighting/detail failures, or a substantial position jump yield uncertainty rather than an invented awake/asleep answer. Missing observations and sampling gaps reset timing rather than treating unseen time as closed.

Engineering defaults: blink duration 0.07–0.60 seconds, prolonged closure 1.5 seconds, maximum observation gap 0.25 seconds. These are tunable demo settings, not clinical thresholds. Low frame rates can miss blinks; the application does not reconstruct missing observations.

The dashboard's 30-second **closed fraction of known time** is accompanied by known-time coverage. It is NOT a validated PERCLOS measure: there is no established 80%-eyelid-occlusion measurement. Low coverage means the ratio is not representative of the entire session.

## Synthetic demo and evaluation

```bash
.venv/bin/python app.py demo
.venv/bin/python app.py evaluate examples/synthetic-labels.csv
.venv/bin/python -m unittest discover -s tests -v
```

The demo produces **one blink and one prolonged-closure event** from manufactured EAR numbers. It is a logic demonstration, not camera accuracy evidence. `docs/demo.json` is its actual generated output.

For a local, constant-frame-rate video you already own:

```bash
.venv/bin/python app.py video input/test.mp4 --no-display --csv outputs/predictions.csv
```

The metrics include `time_s,state,...`. For evaluation, create a separate CSV with `truth,prediction` using independently labelled OPEN/CLOSED truth; map the recorded `state` to `prediction`. There is no automatic ground-truth generation. See [measuring precision](docs/MEASURING_PRECISION.md). Camera export uses the same optional `--csv` argument; output paths must not exist already.

## Limits that must stay visible

Glasses, reflections, partial eyelid visibility, squinting, face shape, camera resolution, fast blinks, downward head angle, and lighting changes can cause mistakes or excessive UNKNOWN output. Geometric gates are heuristics—not calibrated confidence probabilities. Same-location person swaps cannot reliably be detected; keep the session single-person. Recalibrate when the person, camera position, or lighting changes. Local video timestamps use frame index / reported FPS; variable-frame-rate timing is not validated. No sound alarm or background monitoring is implemented.

Read [design](docs/DESIGN.md), [privacy and model handling](docs/PRIVACY_AND_MODELS.md), [validation](docs/VALIDATION.md), and [provenance](docs/PROVENANCE.md). Application code and synthetic examples are MIT-licensed; dependencies/models retain their own terms.

## Environment and installation scope

The verified release baseline uses standard CPython 3.12.

The macOS Apple-Silicon compatibility stack is:

- MediaPipe 0.10.21
- OpenCV contrib 4.11.0
- NumPy 1.26.4

MediaPipe Tasks FaceLandmarker 1.0.1 was rejected for this release because native initialization repeatedly aborted on the maintainer Mac. The release instead uses the legacy Face Mesh backend, which completed creation, blank-image inference, video-mode startup, and the project test suite on the same machine.

Each project uses its own `.venv`; system Python packages are not replaced. Do not use `sudo` for installation.

`pip check` is not used as the macOS release gate because MediaPipe 0.10.21 reports an unsupported-platform metadata warning on this system despite successful native runtime execution. The release gate instead uses exact dependency pins, imports, the project test suite, and real Face Mesh initialization.

Native camera behavior and real-world eye-state accuracy remain separate manual checks.
