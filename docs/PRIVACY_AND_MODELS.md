## Privacy and model provenance

Application code does not upload camera images and camera mode does not record frames.

Calibration is stored under `.local/`; generated reports are stored only when explicitly requested. These paths are ignored by Git. Use only your own image/video or a consenting participant.

The release runtime uses MediaPipe 0.10.21 legacy Face Mesh. The application does not separately download or manage a `.task` model file. Runtime assets are supplied by the installed MediaPipe package and remain subject to MediaPipe's own licensing and package behavior.

Package installation requires network access. Normal application inference does not intentionally open URLs, upload frames, or call an online inference API.

This repository does not claim verified network isolation of third-party native libraries. Do not treat local execution as a formal privacy or security guarantee.
