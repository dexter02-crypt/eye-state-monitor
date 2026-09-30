"""Local I/O and explicit model acquisition. No Git or automatic camera access."""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import urllib.request
import uuid
import zipfile

ROOT = Path(__file__).resolve().parent
MODELS = {
    'face': ('face_landmarker.task', 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'),
    'hand': ('hand_landmarker.task', 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'),
}


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


def download_model(kind: str) -> Path:
    """Explicit HTTPS download; receipts are local TOFU hashes, NOT signed provenance."""
    filename, url = MODELS[kind]
    path = ROOT / 'models' / filename
    receipt = path.with_suffix('.receipt.json')
    if path.exists() or path.is_symlink() or receipt.exists() or receipt.is_symlink():
        print(f'Using existing verified local bytes: {model_path(kind)}')
        return path
    if (ROOT / 'models').is_symlink():
        raise ValueError('Refusing a symlinked model folder.')
    print('Downloading the version-1 model from Google storage. No camera is opened.')
    print('No independently verified upstream SHA-256 was available in this kit.')
    class SameHostRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            from urllib.parse import urlparse
            parsed = urlparse(newurl)
            if parsed.scheme != 'https' or parsed.netloc != 'storage.googleapis.com':
                raise ValueError('Unexpected model-download redirect.')
            return super().redirect_request(req, fp, code, msg, headers, newurl)
    opener = urllib.request.build_opener(SameHostRedirect())
    data = bytearray()
    with opener.open(url, timeout=60) as response:
        while block := response.read(1 << 20):
            data.extend(block)
            if len(data) > 40_000_000:
                raise ValueError('Model download exceeds the 40 MB limit.')
    import io
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if not any(name.endswith('.tflite') for name in archive.namelist()):
                raise ValueError('The download is not the expected task model bundle.')
    except zipfile.BadZipFile as exc:
        raise ValueError('Invalid model bundle received; no model file was saved.') from exc
    digest = hashlib.sha256(data).hexdigest()
    write_new(path, bytes(data))
    write_json(receipt, {'url': url, 'bytes': len(data), 'sha256': digest,
                       'downloaded_utc': datetime.now(timezone.utc).isoformat(),
                       'verification': 'Local trust-on-first-use hash, not an upstream signature'})
    print(f'Model saved: {path}\nSHA-256: {digest}')
    return path


def model_path(kind: str) -> Path:
    filename, url = MODELS[kind]
    path = ROOT / 'models' / filename
    receipt_path = path.with_suffix('.receipt.json')
    if path.is_symlink() or receipt_path.is_symlink() or path.parent.is_symlink():
        raise ValueError('Model path must not be a symlink.')
    if not path.is_file() or path.stat().st_size > 40_000_000:
        raise ValueError('Download the model first: python app.py model')
    receipt = read_json(receipt_path)
    if not isinstance(receipt, dict) or receipt.get('url') != url or receipt.get('bytes') != path.stat().st_size or receipt.get('sha256') != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError('Model differs from its local download receipt; no automatic replacement.')
    return path


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
