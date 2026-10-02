# Changelog

## 0.1.0 — 2026-10-02

Initial public release candidate for Eye State Monitor.

- Tracks calibrated OPEN / CLOSED / ASYMMETRIC / UNKNOWN eye state.
- Counts bounded blink and prolonged-closure events.
- Preserves UNKNOWN states instead of inventing awake/asleep conclusions.
- Uses MediaPipe legacy Face Mesh landmarks with deterministic local EAR logic.
- Supports synthetic evaluation, local video input, and optional local metrics export.
- Does not provide medical, sleep, driving-safety, attendance, or attention conclusions.
