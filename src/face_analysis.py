"""
src/face_analysis.py

Real-time face detection (OpenCV) plus emotion recognition and gender
prediction (DeepFace), exposed as a single `process_frame()` function
that the Gradio webcam stream calls on every frame.

Design notes
------------
* Face *detection* (OpenCV Haar Cascade) is cheap and runs every frame
  so the bounding box tracks the face smoothly.
* Emotion/gender *analysis* (DeepFace) is comparatively expensive, so it
  only runs every `config.EMOTION_GENDER_FRAME_SKIP` frames. Results are
  cached (via Gradio `gr.State`) and reused on the in-between frames so
  the label never disappears, it just updates periodically.
* All failures are caught and converted into the user-facing messages
  required by the spec: "Camera access required." / "Model loading failed."
"""

import logging
import os

import cv2
import numpy as np

import config
from src.utils import draw_multiline_tags, message_frame

logger = logging.getLogger("ai_vision_studio.face_analysis")

_DEEPFACE_IMPORT_ERROR = None
try:
    from deepface import DeepFace
except Exception as exc:  # pragma: no cover - exercised only if dependency missing
    DeepFace = None
    _DEEPFACE_IMPORT_ERROR = exc


class FaceAnalyzer:
    """Loads the OpenCV face detector and warms up DeepFace once, then
    exposes fast per-frame detection + analysis methods."""

    def __init__(self):
        self.ready = False
        self.error = None
        self.face_cascade = None

        try:
            cascade_path = os.path.join(
                cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
            )
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                raise RuntimeError("Failed to load OpenCV Haar cascade for face detection.")

            if DeepFace is None:
                raise RuntimeError(f"DeepFace is unavailable: {_DEEPFACE_IMPORT_ERROR}")

            # Warm up DeepFace's emotion + gender models once at startup so
            # weight downloads / graph building don't stall the first
            # webcam frame the user sees.
            dummy = np.full((120, 120, 3), 128, dtype=np.uint8)
            DeepFace.analyze(
                dummy,
                actions=["emotion", "gender"],
                detector_backend="skip",
                enforce_detection=False,
                silent=True,
            )
            self.ready = True
        except Exception as exc:
            logger.exception("FaceAnalyzer failed to initialize")
            self.error = str(exc)
            self.ready = False

    def detect_faces(self, gray_frame):
        """Returns a list of (x, y, w, h) boxes in pixel coordinates."""
        return self.face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60)
        )

    def analyze_face(self, face_crop_bgr):
        """Runs DeepFace emotion + gender analysis on a single cropped
        face image. Returns a dict or None on failure."""
        try:
            result = DeepFace.analyze(
                face_crop_bgr,
                actions=["emotion", "gender"],
                detector_backend="skip",   # we already cropped the face ourselves
                enforce_detection=False,
                silent=True,
            )
            if isinstance(result, list):
                result = result[0]

            emotion_scores = result.get("emotion", {}) or {}
            dominant_emotion = result.get("dominant_emotion", "neutral")
            emotion_conf = float(emotion_scores.get(dominant_emotion, 0.0))

            gender_scores = result.get("gender", {}) or {}
            if gender_scores:
                top_gender = max(gender_scores, key=gender_scores.get)
                gender_conf = float(gender_scores[top_gender])
            else:
                top_gender, gender_conf = "Unknown", 0.0

            return {
                "emotion": dominant_emotion.capitalize(),
                "emotion_conf": emotion_conf,
                "gender": "Woman" if top_gender.lower().startswith("w") else "Man",
                "gender_conf": gender_conf,
            }
        except Exception:
            logger.exception("DeepFace analysis failed on a face crop")
            return None


# Module-level singleton so the (relatively slow) model warm-up only
# happens once per process, not once per webcam frame.
_analyzer = None


def get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = FaceAnalyzer()
    return _analyzer


def _format_gender_label(info):
    if info["gender"] == "Analyzing...":
        return "Analyzing..."
    if info["gender_conf"] <= 0:
        return "Prediction Uncertain"
    if info["gender_conf"] < config.GENDER_UNCERTAIN_THRESHOLD_PCT:
        return "Prediction Uncertain"
    return f"{info['gender']} ({info['gender_conf']:.0f}%)"


def _format_emotion_label(info):
    if info["emotion"] == "Analyzing...":
        return "Analyzing..."
    return f"{info['emotion']} ({info['emotion_conf']:.0f}%)"


def process_frame(rgb_frame, frame_count, face_cache):
    """Main per-frame entry point used by the Gradio `.stream()` callback.

    Parameters
    ----------
    rgb_frame : np.ndarray | None
        RGB frame from the Gradio webcam component. None if the camera
        is unavailable / permission was denied.
    frame_count : int
        Persisted across calls via gr.State.
    face_cache : dict
        Persisted across calls via gr.State. Caches the last known
        emotion/gender result per detected-face slot.

    Returns
    -------
    (annotated_rgb_frame, frame_count, face_cache)
    """
    if rgb_frame is None:
        return (
            message_frame("Camera access required.", sub_message="Please allow webcam access in your browser."),
            frame_count,
            face_cache,
        )

    analyzer = get_analyzer()
    if not analyzer.ready:
        return (
            message_frame("Model loading failed.", sub_message=str(analyzer.error)[:90]),
            frame_count,
            face_cache,
        )

    frame_count = (frame_count or 0) + 1
    face_cache = face_cache or {}

    try:
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = analyzer.detect_faces(gray)
    except Exception:
        logger.exception("Face detection failed on this frame")
        return (
            message_frame("Model loading failed.", sub_message="Face detector error on this frame."),
            frame_count,
            face_cache,
        )

    run_deepface = (frame_count % config.EMOTION_GENDER_FRAME_SKIP == 0) or (len(face_cache) == 0)
    new_cache = {}

    for idx, (x, y, w, h) in enumerate(faces):
        key = f"face_{idx}"
        pad_x, pad_y = int(w * 0.12), int(h * 0.12)
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(bgr.shape[1], x + w + pad_x), min(bgr.shape[0], y + h + pad_y)
        crop = bgr[y1:y2, x1:x2]

        info = None
        if run_deepface and crop.size > 0:
            info = analyzer.analyze_face(crop)

        if info is None:
            info = face_cache.get(
                key,
                {"emotion": "Analyzing...", "emotion_conf": 0.0,
                 "gender": "Analyzing...", "gender_conf": 0.0},
            )

        new_cache[key] = info

        emotion_label = _format_emotion_label(info)
        gender_label = _format_gender_label(info)
        box_color = config.EMOTION_COLORS.get(info["emotion"].lower(), (70, 160, 255))

        draw_multiline_tags(
            bgr,
            (x1, y1, x2, y2),
            [(emotion_label, box_color), (gender_label, config.GENDER_TAG_COLOR)],
            base_color=box_color,
        )

    if len(faces) == 0:
        cv2.putText(bgr, "No face detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2, cv2.LINE_AA)

    out_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return out_rgb, frame_count, new_cache
