# Validation scope — 2 October 2026

## Verified local compatibility

The release candidate was exercised on the maintainer Apple-Silicon Mac using:

- CPython 3.12
- MediaPipe 0.10.21
- OpenCV contrib 4.11.0
- NumPy 1.26.4

The MediaPipe legacy Face Mesh backend successfully:

- initialized,
- closed cleanly,
- performed blank-image inference with zero detected faces,
- started video mode,
- and ran inside the project's native backend test.

The synthetic demonstration produced exactly one blink and one prolonged-closure event.

## Rejected runtime path

MediaPipe Tasks FaceLandmarker 1.0.1 repeatedly aborted during native initialization on the same Mac, including in clean Python 3.12 and Python 3.14 environments.

The failure occurred in the native MediaPipe/Drishti path before useful inference. The release therefore does not use that Tasks backend.

## Test scope

The release candidate exercises:

- EAR geometry,
- scale/translation/rotation properties,
- calibration separation and refusal cases,
- hysteresis,
- blink timing,
- prolonged-closure timing,
- sampling-gap resets,
- UNKNOWN and ASYMMETRIC behavior,
- time-weighted coverage,
- evaluation metrics,
- exclusive local outputs,
- synthetic CLI demonstration,
- missing-profile failure,
- release-version identity,
- and real native Face Mesh blank-image inference.

These results do not establish webcam accuracy, medical validity, sleep classification, attention measurement, or driving-safety fitness.

## Remaining manual boundary

Physical webcam calibration and live eye-state behavior remain manual checks.

Use only the maintainer or another consenting participant. Do not publish personal calibration profiles or recordings.
