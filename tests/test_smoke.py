"""
tests/test_smoke.py

Lightweight smoke tests that don't require downloading model weights, so
they run quickly in CI. They check that:
    * core modules import cleanly
    * drawing/utility helpers behave as expected
    * config values are well-formed

Run with:  pytest tests/
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config  # noqa: E402
from src.utils import color_from_label, draw_label_box, message_frame  # noqa: E402


def test_config_values_well_formed():
    assert config.DEFAULT_DETECTION_ENGINE in config.DETECTION_ENGINES
    assert 0.0 < config.DEFAULT_CONFIDENCE_THRESHOLD < 1.0
    assert len(config.OPEN_VOCAB_OBJECTS) > 20
    assert set(config.EMOTIONS) == set(config.EMOTION_COLORS.keys())


def test_color_from_label_is_deterministic():
    c1 = color_from_label("phone")
    c2 = color_from_label("phone")
    c3 = color_from_label("book")
    assert c1 == c2
    assert all(0 <= v <= 255 for v in c1)
    assert c1 != c3


def test_draw_label_box_returns_same_shape():
    frame = np.full((200, 300, 3), 255, dtype=np.uint8)
    out = draw_label_box(frame, (10, 10, 100, 100), "Phone (95%)", color=(70, 160, 255))
    assert out.shape == frame.shape


def test_message_frame_has_requested_size():
    frame = message_frame("Camera access required.", width=320, height=240)
    assert frame.shape == (240, 320, 3)
    assert frame.dtype == np.uint8


def test_object_detection_module_imports():
    # Import only — does not instantiate any model, so no network/weights
    # download is required for this test to pass.
    import src.object_detection as od  # noqa: F401
    assert hasattr(od, "get_detector")
    assert hasattr(od, "process_frame")


def test_face_analysis_module_imports():
    import src.face_analysis as fa  # noqa: F401
    assert hasattr(fa, "get_analyzer")
    assert hasattr(fa, "process_frame")
