import cv2, numpy as np
import mediapipe as mp
from pathlib import Path

FACE_MODEL = "selfie_prototype/face_landmarker.task"
IMG_SIZE = 224
MARGIN = 2.5
MIN_FACE_FRAC = 0.20
FRONTAL = (0.30, 0.70)

_lm = None
def _landmarker():
    global _lm
    if _lm is None:
        if not Path(FACE_MODEL).exists():
            raise FileNotFoundError(f"Face model not found at {Path(FACE_MODEL).resolve()}")
        _lm = mp.tasks.vision.FaceLandmarker.create_from_options(mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=FACE_MODEL),
            running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=2))
    return _lm


def check_face(img_bgr):
    """The face gate. Returns (ok, reason, box) where box = (x0, y0, x1, y1) in pixels."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = _landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    if not res.face_landmarks:
        return False, "No face found", None
    if len(res.face_landmarks) > 1:
        return False, "More than one face", None
    pts = np.array([[p.x, p.y] for p in res.face_landmarks[0]])          # 0-1 coordinates
    if pts.min() < -0.02 or pts.max() > 1.02:
        return False, "Face is cut off by the edge of the photo", None
    h, w = img_bgr.shape[:2]
    x0, y0 = pts.min(0) * (w, h)
    x1, y1 = pts.max(0) * (w, h)
    if (x1 - x0) / w < MIN_FACE_FRAC:
        return False, "Face is too small - move closer", None
    ratio = (pts[1][0] - pts[234][0]) / (pts[454][0] - pts[234][0] + 1e-6)   # nose between the two cheek points
    if not FRONTAL[0] <= ratio <= FRONTAL[1]:
        return False, "Face is not looking at the camera", None
    return True, "OK", (int(x0), int(y0), int(x1), int(y1))


def crop_face(img_bgr, box):
    """Square crop around the face (MARGIN x the face box), resized to IMG_SIZE x IMG_SIZE."""
    x0, y0, x1, y1 = box
    cx, cy, side = (x0 + x1) // 2, (y0 + y1) // 2, int(max(x1 - x0, y1 - y0) * MARGIN)
    big = cv2.copyMakeBorder(img_bgr, side, side, side, side, cv2.BORDER_CONSTANT)   # so the crop never leaves the photo
    sx, sy = cx - side // 2 + side, cy - side // 2 + side
    return cv2.resize(big[sy:sy + side, sx:sx + side], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def prepare(img_bgr):
    """Gate + crop in one call. Returns (ok, reason, crop_or_None)."""
    ok, reason, box = check_face(img_bgr)
    return ok, reason, (crop_face(img_bgr, box) if ok else None)

def normalize_lighting(crop_bgr):
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)   # evens out local brightness/shadow
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def spoof_probability(model, crop_bgr):
    """P(spoof) for a face crop. The model expects RGB with values 0-255 (it rescales inside)."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype("float32")
    return float(model.predict(rgb[None], verbose=0)[0][0]) 

