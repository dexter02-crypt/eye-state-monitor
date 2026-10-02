#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

"$PYTHON" -c '
import sys
assert sys.version_info[:2] == (3, 12), \
    "This release baseline requires standard CPython 3.12."
'

if [ -L .venv ]; then
  echo "Refusing a symlinked virtual environment." >&2
  exit 2
fi

if [ ! -e .venv ]; then
  "$PYTHON" -m venv .venv
fi

.venv/bin/python -c '
import sys
assert sys.version_info[:2] == (3, 12), \
    "Existing .venv is not Python 3.12."
'

.venv/bin/python -m pip install \
  --only-binary=:all: \
  -r requirements.txt

.venv/bin/python - <<'PY'
from importlib.metadata import distributions

names = {
    d.metadata["Name"].lower()
    for d in distributions()
}

assert "mediapipe" in names
assert "opencv-contrib-python" in names
assert "opencv-python" not in names
assert "opencv-python-headless" not in names
assert "opencv-contrib-python-headless" not in names

import cv2
import mediapipe
import numpy

print("MediaPipe:", mediapipe.__version__)
print("OpenCV:", cv2.__version__)
print("NumPy:", numpy.__version__)
PY

.venv/bin/python -B \
  -m unittest discover -s tests -v

.venv/bin/python app.py check

echo "Setup, tests, and native Face Mesh smoke check finished."
