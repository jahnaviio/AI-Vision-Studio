"""
src/utils.py

Shared drawing and helper utilities used by both the Face Analysis and
Object Detection pipelines: bounding-box rendering, text tags, label
coloring, and the "error frame" placeholders shown when the camera is
unavailable or a model fails to load.
"""

import hashlib

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def color_from_label(label: str):
    """Deterministically map a text label to a vivid BGR color so the
    same object/class always gets the same box color across frames."""
    h = hashlib.md5(label.encode("utf-8")).hexdigest()
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (b % 200 + 35, g % 200 + 35, r % 200 + 35)


def draw_label_box(frame, box, label, color=(70, 160, 255), thickness=2):
    """Draw a single bounding box with a filled label tag above it.

    box: (x1, y1, x2, y2) in pixel coordinates (any numeric type).
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    if label:
        (text_w, text_h), baseline = cv2.getTextSize(label, FONT, 0.55, 1)
        label_y1 = max(0, y1 - text_h - baseline - 6)
        cv2.rectangle(frame, (x1, label_y1), (x1 + text_w + 8, y1), color, -1)
        text_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
        cv2.putText(frame, label, (x1 + 4, y1 - 5), FONT, 0.55, text_color, 1, cv2.LINE_AA)
    return frame


def draw_multiline_tags(frame, box, lines, base_color=(70, 160, 255)):
    """Draw a bounding box plus several stacked tags above it.

    `lines` is a list of (text, bgr_color) tuples, drawn bottom-to-top
    stacked above the box. Used for Face Analysis (Emotion + Gender).
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), base_color, 2)

    y_cursor = y1
    for line, color in reversed(lines):
        (text_w, text_h), baseline = cv2.getTextSize(line, FONT, 0.55, 1)
        y_top = max(0, y_cursor - text_h - baseline - 6)
        cv2.rectangle(frame, (x1, y_top), (x1 + text_w + 10, y_cursor), color, -1)
        text_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
        cv2.putText(frame, line, (x1 + 5, y_cursor - 5), FONT, 0.55, text_color, 1, cv2.LINE_AA)
        y_cursor = y_top
    return frame


def message_frame(message: str, width: int = 640, height: int = 480, sub_message: str = ""):
    """Create a plain white frame with a centered black message.

    Used for the required error states:
        "Camera access required."
        "Model loading failed."
    """
    frame = np.full((height, width, 3), 255, dtype=np.uint8)
    (text_w, text_h), _ = cv2.getTextSize(message, FONT, 0.9, 2)
    x = max(10, (width - text_w) // 2)
    y = height // 2 if not sub_message else height // 2 - 10
    cv2.putText(frame, message, (x, y), FONT, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

    if sub_message:
        (sw, sh), _ = cv2.getTextSize(sub_message, FONT, 0.55, 1)
        sx = max(10, (width - sw) // 2)
        cv2.putText(frame, sub_message, (sx, y + 35), FONT, 0.55, (90, 90, 90), 1, cv2.LINE_AA)

    return frame
