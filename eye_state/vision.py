"""MediaPipe adapter. Quality gates are heuristics, not calibrated confidence scores."""
from __future__ import annotations
import math
from local_support import model_path, validate_frame
from .core import eye_aspect_ratio
LEFT = (362, 385, 387, 263, 373, 380)
RIGHT = (33, 160, 158, 133, 153, 144)


def create_detector(video=False):
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path('face'))),
        running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
        num_faces=2, min_face_detection_confidence=.7,
        min_face_presence_confidence=.7, min_tracking_confidence=.7)
    return vision.FaceLandmarker.create_from_options(options)


def infer(detector, frame, timestamp_ms=None):
    import cv2
    import mediapipe as mp
    image = mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(validate_frame(frame), cv2.COLOR_BGR2RGB))
    return detector.detect(image) if timestamp_ms is None else detector.detect_for_video(image, timestamp_ms)


def measure(faces, width, height):
    """Return left EAR, right EAR, reason, eye contours; require one visible frontal face."""
    if len(faces) != 1:
        return None, None, 'NO_FACE' if not faces else 'MULTIPLE_FACES', []
    landmarks = faces[0]
    if len(landmarks) < 468:
        return None, None, 'INCOMPLETE_LANDMARKS', []
    try:
        contours = [[(landmarks[i].x*width, landmarks[i].y*height) for i in indices] for indices in (LEFT,RIGHT)]
        all_points = [p for group in contours for p in group]
        if any(not math.isfinite(v) for p in all_points for v in p):
            raise ValueError('Non-finite point')
        if any(not (0 <= x < width and 0 <= y < height) for x,y in all_points):
            return None, None, 'EYES_OUT_OF_FRAME', contours
        widths = [math.dist(p[0],p[3]) for p in contours]
        if min(widths) < 18:
            return None, None, 'TOO_FAR_AWAY', contours
        if min(widths)/max(widths) < .6:
            return None, None, 'TURN_TOWARD_CAMERA', contours
        centers = [((p[0][0]+p[3][0])/2, (p[0][1]+p[3][1])/2) for p in contours]
        dx, dy = centers[0][0]-centers[1][0], centers[0][1]-centers[1][1]
        if abs(math.degrees(math.atan2(dy,abs(dx)))) > 25:
            return None, None, 'HEAD_TILT', contours
        distance = math.dist(*centers)
        if distance < 40:
            return None, None, 'TOO_FAR_AWAY', contours
        nose_x = landmarks[1].x*width
        if not math.isfinite(nose_x) or abs(nose_x-(centers[0][0]+centers[1][0])/2)/distance > .30:
            return None, None, 'TURN_TOWARD_CAMERA', contours
        values = [eye_aspect_ratio(p) for p in contours]
        if any(not 0 <= value <= .8 for value in values):
            return None, None, 'IMPLAUSIBLE_GEOMETRY', contours
        return values[0], values[1], '', contours
    except (ValueError, TypeError, AttributeError, IndexError):
        return None, None, 'INVALID_LANDMARKS', []


def measure_frame(result, frame):
    import cv2
    validate_frame(frame)
    left, right, reason, contours = measure(result.face_landmarks, frame.shape[1], frame.shape[0])
    if not reason:
        all_points = [p for group in contours for p in group]
        x0,y0 = max(0,int(min(p[0] for p in all_points))-5), max(0,int(min(p[1] for p in all_points))-10)
        x1,y1 = min(frame.shape[1],int(max(p[0] for p in all_points))+6), min(frame.shape[0],int(max(p[1] for p in all_points))+11)
        gray = cv2.cvtColor(frame[y0:y1,x0:x1], cv2.COLOR_BGR2GRAY)
        if gray.size == 0 or gray.mean() < 25 or gray.mean() > 235:
            return None,None,'POOR_LIGHTING',contours
        if cv2.Laplacian(gray, cv2.CV_64F).var() < 12:
            return None,None,'BLUR_OR_LOW_DETAIL',contours
    return left,right,reason,contours
