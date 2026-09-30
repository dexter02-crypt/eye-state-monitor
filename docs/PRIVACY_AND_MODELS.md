## Privacy and model provenance

Application code does not upload camera images. Camera mode does not record frames. Calibration is stored under `.local/`, model files under `models/`, and generated reports under `outputs/`; these folders are ignored by Git and excluded by the kit publisher. Eye metrics are saved only with an explicit `--csv` argument. Use only your own image or a consenting participant; do not publish personal recordings or calibration profiles.

This is not a promise of zero networking: installation and explicit model acquisition require network access. MediaPipe's published privacy notice says input processing is on-device but its Tasks APIs send performance/utilization metrics to Google. Review that notice before use: https://pypi.org/project/mediapipe/#privacy-notice . The package has no telemetry-disable implementation or verified network isolation.

`model` explicitly downloads the version-1 `.task` bundle from the official Google storage endpoint and records the received bytes' SHA-256. Subsequent loads compare the file to that local receipt. This is trust-on-first-use, NOT an independently verified upstream signature. The model weights are not bundled, not MIT-relicensed by this project, and not uploaded by the publishing helper. A hash disagreement stops instead of silently replacing files.
