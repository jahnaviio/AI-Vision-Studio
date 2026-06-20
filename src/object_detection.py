"""
src/object_detection.py

Open-vocabulary object detection with three interchangeable backends:

    * YOLO-World     -> fast, real-time CPU/GPU detection (DEFAULT)
    * Grounding DINO -> transformer-based zero-shot detection, slower, very
                        accurate, best for static / lower frame-rate use
    * Florence-2     -> Microsoft's unified vision-language model, used
                        here for open-vocabulary detection via its
                        <OD> object-detection task

All three implement the same informal interface:

    detect(bgr_frame, prompts, confidence) -> list[(box_xyxy, label, score)]

Models are loaded lazily (only when first selected) and then cached, so
starting the app doesn't pay the cost of loading every backend up front.
"""

import logging

import cv2
import numpy as np
from PIL import Image

import config
from src.utils import color_from_label, draw_label_box, message_frame

logger = logging.getLogger("ai_vision_studio.object_detection")


# ----------------------------------------------------------------------------
# YOLO-World — default real-time engine
# ----------------------------------------------------------------------------
class YOLOWorldDetector:
    """Open-vocabulary detector via Ultralytics YOLO-World.

    Fast enough for genuine real-time webcam use on CPU. Classes are set
    dynamically from a free-text prompt list, which is what gives it
    open-vocabulary behaviour beyond the fixed 80 COCO classes.
    """

    name = "YOLO-World"

    def __init__(self, weights: str = config.YOLO_WORLD_WEIGHTS):
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(f"ultralytics import failed: {exc}") from exc

        try:
            self.model = YOLO(weights)
        except Exception as exc:
            raise RuntimeError(f"Could not load YOLO-World weights '{weights}': {exc}") from exc

        self.classes = []
        self.set_classes(config.OPEN_VOCAB_OBJECTS)

    def set_classes(self, classes):
        classes = [c for c in classes if c] or config.OPEN_VOCAB_OBJECTS
        self.classes = classes
        self.model.set_classes(classes)

    def detect(self, bgr_frame, prompts=None, confidence=0.25):
        if prompts and set(prompts) != set(self.classes):
            self.set_classes(prompts)

        results = self.model.predict(bgr_frame, conf=confidence, verbose=False)[0]
        detections = []
        for box in results.boxes:
            xyxy = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            score = float(box.conf[0])
            label = self.classes[cls_id] if cls_id < len(self.classes) else str(cls_id)
            detections.append((xyxy, label, score))
        return detections


# ----------------------------------------------------------------------------
# Grounding DINO — high-accuracy zero-shot text-grounded detection
# ----------------------------------------------------------------------------
class GroundingDINODetector:
    """Open-vocabulary detector via HuggingFace `transformers`
    zero-shot-object-detection (Grounding DINO). More accurate at
    grounding free-text phrases than YOLO-World, but noticeably slower
    on CPU — best used for higher-accuracy / lower frame-rate sessions.
    """

    name = "Grounding DINO"

    def __init__(self, model_id: str = config.GROUNDING_DINO_MODEL_ID):
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        except Exception as exc:
            raise RuntimeError(f"transformers/torch import failed: {exc}") from exc

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
            self.model.eval()
        except Exception as exc:
            raise RuntimeError(f"Could not load Grounding DINO '{model_id}': {exc}") from exc

    def detect(self, bgr_frame, prompts=None, confidence=0.25):
        prompts = prompts or config.OPEN_VOCAB_OBJECTS
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        text_prompt = ". ".join(prompts) + "."

        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=confidence,
            text_threshold=confidence,
            target_sizes=[image.size[::-1]],
        )[0]

        detections = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            label = label.strip() or "object"
            detections.append((box.tolist(), label, float(score)))
        return detections


# ----------------------------------------------------------------------------
# Florence-2 — unified vision-language model used for open detection
# ----------------------------------------------------------------------------
class Florence2Detector:
    """Open-vocabulary-ish detector via Microsoft Florence-2's built-in
    `<OD>` object detection task. Florence-2 does not emit confidence
    scores for plain object detection, so detections are shown without
    a percentage in the UI. Slowest of the three engines on CPU.
    """

    name = "Florence-2"

    def __init__(self, model_id: str = config.FLORENCE2_MODEL_ID):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except Exception as exc:
            raise RuntimeError(f"transformers/torch import failed: {exc}") from exc

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=self.dtype,
                attn_implementation="eager",
            ).to(self.device).eval()
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        except Exception as exc:
            raise RuntimeError(f"Could not load Florence-2 '{model_id}': {exc}") from exc

    def detect(self, bgr_frame, prompts=None, confidence=0.25):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        task = "<OD>"

        inputs = self.processor(text=task, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device, self.dtype) if v.dtype.is_floating_point else v.to(self.device)
                  for k, v in inputs.items()}

        with self._torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=1,
                do_sample=False,
            )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            generated_text, task=task, image_size=(image.width, image.height)
        )
        od = parsed.get(task, {})
        bboxes = od.get("bboxes", [])
        labels = od.get("labels", [])

        # Florence-2's plain <OD> task does not return scores; we report
        # them as None and the UI renders the label without a percentage.
        return [(box, label, None) for box, label in zip(bboxes, labels)]


# ----------------------------------------------------------------------------
# Factory + frame pipeline
# ----------------------------------------------------------------------------
_detector_cache = {}


def get_detector(engine_name: str):
    """Lazily instantiate and cache a detector backend by name."""
    if engine_name in _detector_cache and _detector_cache[engine_name] is not None:
        return _detector_cache[engine_name]

    if engine_name == "YOLO-World":
        detector = YOLOWorldDetector()
    elif engine_name == "Grounding DINO":
        detector = GroundingDINODetector()
    elif engine_name == "Florence-2":
        detector = Florence2Detector()
    else:
        raise ValueError(f"Unknown detection engine: {engine_name}")

    _detector_cache[engine_name] = detector
    return detector


def process_frame(rgb_frame, engine_name, prompt_text, confidence, frame_count):
    """Main per-frame entry point used by the Gradio `.stream()` callback.

    Returns (annotated_rgb_frame, frame_count).
    """
    if rgb_frame is None:
        return (
            message_frame("Camera access required.", sub_message="Please allow webcam access in your browser."),
            frame_count,
        )

    frame_count = (frame_count or 0) + 1
    prompts = [p.strip() for p in (prompt_text or "").split(",") if p.strip()]
    if not prompts:
        prompts = config.OPEN_VOCAB_OBJECTS

    try:
        detector = get_detector(engine_name)
    except Exception as exc:
        logger.exception("Failed to load detection engine '%s'", engine_name)
        return (
            message_frame("Model loading failed.", sub_message=str(exc)[:90]),
            frame_count,
        )

    bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    try:
        detections = detector.detect(bgr, prompts=prompts, confidence=confidence)
    except Exception:
        logger.exception("Detection failed on this frame with engine '%s'", engine_name)
        return (
            message_frame("Model loading failed.", sub_message="Detection error on this frame."),
            frame_count,
        )

    for box, label, score in detections:
        tag = label.title() if score is None else f"{label.title()} ({score * 100:.0f}%)"
        color = color_from_label(label)
        draw_label_box(bgr, box, tag, color=color)

    if len(detections) == 0:
        cv2.putText(bgr, "No objects detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2, cv2.LINE_AA)

    out_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return out_rgb, frame_count
