"""Local I/O helpers. No Git or automatic camera access."""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parent

def finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number.')
    return float(value)


def write_new(path: Path, data: bytes) -> None:
    """Never overwrite a file, including a symlink. New files are owner-readable only."""
    path = Path(path)
    if any(p.is_symlink() for p in [path, *path.parents]):
        # macOS /var and /tmp may be system symlinks; user-owned output folders
        # under Downloads do not need those aliases. Resolve those parents explicitly.
        raise ValueError('Output path contains a symlink; use a real directory.')
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data)


def write_json(path: Path, data: object) -> None:
    write_new(path, (json.dumps(data, indent=2, allow_nan=False) + '\n').encode())


def unique_output(suffix: str) -> Path:
    name = datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:8]
    return ROOT / 'outputs' / (name + suffix)


def read_json(path: Path, limit: int = 1_000_000) -> object:
    if not path.is_file() or path.stat().st_size > limit:
        raise ValueError('Missing or oversized JSON file.')
    return json.loads(path.read_text(encoding='utf-8'),
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError('Non-finite JSON value')))


@contextmanager
def capture(camera: int = 0, video: str | None = None):
    import cv2
    if isinstance(camera, bool) or not isinstance(camera, int) or camera < 0:
        raise ValueError('Camera index must be a non-negative integer.')
    if video is not None:
        source = Path(video).expanduser()
        if not source.is_file() or source.stat().st_size > 2_000_000_000:
            raise ValueError('Video must be an existing local file of at most 2 GB, not a URL.')
        cap = cv2.VideoCapture(str(source))
    else:
        cap = cv2.VideoCapture(camera, cv2.CAP_AVFOUNDATION if sys.platform == 'darwin' else cv2.CAP_ANY)
    try:
        if not cap.isOpened():
            raise ValueError('Cannot open camera/video. On Mac allow Camera access for Terminal, then retry. No recording is made.')
        if video is None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        yield cap
    finally:
        cap.release()


def validate_frame(frame):
    import numpy as np
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 16 or frame.shape[0]*frame.shape[1] > 40_000_000:
        raise ValueError('Expected a BGR uint8 image, at least 16x16 and at most 40 megapixels.')
    return frame


def load_image(path: str):
    import cv2
    target = Path(path).expanduser()
    if not target.is_file() or target.stat().st_size > 30_000_000:
        raise ValueError('Image must be an existing local file of at most 30 MB.')
    return validate_frame(cv2.imread(str(target), cv2.IMREAD_COLOR))


def save_image(path: Path, frame):
    import cv2
    validate_frame(frame)
    if path.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
        raise ValueError('Image output must be PNG or JPEG.')
    ok, encoded = cv2.imencode(path.suffix, frame)
    if not ok:
        raise ValueError('Image encoding failed.')
    write_new(path, encoded.tobytes())


def overlay(frame, lines):
    """Plain OpenCV dashboard, dynamically sized; modifies only the display copy."""
    import cv2
    import numpy as np
    width = max(900, frame.shape[1])
    header = np.full((38 + 28 * len(lines), width, 3), 20, dtype=np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(header, str(text)[:110], (16, 30+28*i), cv2.FONT_HERSHEY_SIMPLEX, .57, (235,235,235), 1, cv2.LINE_AA)
    if frame.shape[1] < width:
        frame = cv2.copyMakeBorder(frame, 0, 0, (width-frame.shape[1])//2,
                                  width-frame.shape[1]-(width-frame.shape[1])//2, cv2.BORDER_CONSTANT)
    return np.vstack([header, frame])
