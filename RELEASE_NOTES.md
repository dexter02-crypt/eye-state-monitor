# eye-state-monitor 0.1.0

Eye State Monitor is a local computer-vision experiment for calibrated OPEN / CLOSED / ASYMMETRIC / UNKNOWN eye state, blink events, and prolonged eye closure.

It is not a sleep detector, medical device, driver-safety alarm, attendance system, or proof of attention.

## Release runtime

The verified macOS release baseline uses:

- CPython 3.12
- MediaPipe 0.10.21 legacy Face Mesh
- OpenCV contrib 4.11.0
- NumPy 1.26.4

A previously prepared MediaPipe Tasks FaceLandmarker 1.0.1 path was rejected after repeatable native startup crashes on the maintainer Apple-Silicon Mac.

## Validation

The legacy Face Mesh backend successfully initialized, closed, processed a blank image, and started video mode locally.

The release test suite includes real native Face Mesh startup/inference in addition to deterministic geometry, temporal-state, evaluation, and I/O tests.

The synthetic demo produces one blink and one prolonged-closure event from manufactured EAR values.

No real-world accuracy percentage is claimed.

## Privacy and scope

Camera frames are processed locally and are not recorded by default.

Calibration and optional metrics remain local.

Third-party MediaPipe/OpenCV runtime assets retain their own licensing and behavior.

This release makes no medical, sleep, driving-safety, or attention guarantees.
