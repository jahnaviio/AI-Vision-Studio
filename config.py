"""
config.py

Global configuration values for AI Vision Studio.
Centralizing these makes it easy to tune performance and behavior
without touching the application or model code.
"""

# ----------------------------------------------------------------------------
# Face Analysis (OpenCV + DeepFace)
# ----------------------------------------------------------------------------

# How often (in frames) to run the heavier DeepFace emotion/gender analysis.
# Face *detection* (cheap, OpenCV) still runs every frame for a smooth box,
# but emotion/gender (expensive) is refreshed every Nth frame and cached
# in between so the app stays responsive on CPU.
EMOTION_GENDER_FRAME_SKIP = 4

# Below this gender confidence (%), the UI shows "Prediction Uncertain"
# instead of a Man/Woman label.
GENDER_UNCERTAIN_THRESHOLD_PCT = 60.0

EMOTIONS = ["happy", "sad", "angry", "fear", "surprise", "neutral", "disgust"]

# BGR colors used to tint each emotion's bounding box / tag.
EMOTION_COLORS = {
    "happy":    (60, 200, 60),
    "sad":      (200, 120, 40),
    "angry":    (40, 40, 220),
    "fear":     (160, 40, 160),
    "surprise": (40, 200, 220),
    "neutral":  (160, 160, 160),
    "disgust":  (40, 140, 80),
}

GENDER_TAG_COLOR = (230, 170, 60)  # BGR, light blue-ish accent for gender tag

# ----------------------------------------------------------------------------
# Object Detection (YOLO-World / Grounding DINO / Florence-2)
# ----------------------------------------------------------------------------

DEFAULT_DETECTION_ENGINE = "YOLO-World"
DETECTION_ENGINES = ["YOLO-World", "Grounding DINO", "Florence-2"]

DEFAULT_CONFIDENCE_THRESHOLD = 0.25

# Ultralytics auto-downloads this checkpoint on first use.
YOLO_WORLD_WEIGHTS = "yolov8s-worldv2.pt"

# HuggingFace model IDs (downloaded + cached automatically by `transformers`).
GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
FLORENCE2_MODEL_ID = "microsoft/Florence-2-base"

# A broad open-vocabulary list covering common household / office / desk
# items. Goes well beyond the standard 80 COCO classes used by plain YOLOv8.
OPEN_VOCAB_OBJECTS = [
    "pen", "pencil", "eraser", "sharpener", "notebook", "book",
    "mobile phone", "smartphone", "charger", "cable", "earbuds",
    "headphones", "keyboard", "mouse", "laptop", "monitor", "fan",
    "air conditioner", "refrigerator", "bottle", "cup", "mug",
    "remote", "remote control", "backpack", "bag", "clock", "watch",
    "microphone", "webcam", "speaker", "chair", "table", "lamp",
    "glasses", "spectacles", "wallet", "key", "scissors", "stapler",
    "calculator", "tablet", "tv", "television", "door", "window",
    "plant", "bed", "pillow", "shoe", "shoes", "umbrella", "mask",
    "id card", "paper", "file folder", "box", "tape", "person",
]

# ----------------------------------------------------------------------------
# General App Settings
# ----------------------------------------------------------------------------

APP_TITLE = "AI Vision Studio"
APP_SUBTITLE = "Real-Time Face Analysis and Object Detection"

# How frequently (seconds) Gradio pulls a new webcam frame for processing.
# Lower = more responsive but more CPU/GPU load. 0.1-0.15 is a good balance
# for CPU-only Spaces.
STREAM_EVERY_SECONDS = 0.12
