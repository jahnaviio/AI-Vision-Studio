"""
app.py

AI Vision Studio — main entry point.

A real-time, webcam-driven computer vision app with two modes:
    1. Face Analysis      -> face detection + emotion + gender (live)
    2. Object Detection   -> open-vocabulary object detection (live)

This is a live-stream app, not an image-upload tool: as soon as the user
grants camera access, frames are continuously pulled from the webcam and
piped through the relevant analysis pipeline via Gradio's `.stream()` API.

Run locally:
    python app.py

Deploy:
    Push this repo to a Hugging Face Space with SDK = gradio (see README.md).
"""

import logging

import gradio as gr

import config
from src import face_analysis, object_detection
from src.utils import message_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ai_vision_studio.app")


# ----------------------------------------------------------------------------
# Simple, student-project styling: white background, black text,
# light-blue buttons. No animations, no dashboard chrome.
# ----------------------------------------------------------------------------
CUSTOM_CSS = """
.gradio-container { background-color: #FFFFFF !important; }
body { background-color: #FFFFFF !important; }

.av-title {
    color: #000000 !important;
    text-align: center;
    font-weight: 700;
    margin-bottom: 0.25rem;
}
.av-subtitle {
    color: #333333 !important;
    text-align: center;
    margin-bottom: 1.25rem;
}
.av-note {
    color: #555555 !important;
    text-align: center;
    font-size: 0.92rem;
}

/* Light blue primary buttons */
.av-btn, .av-btn button {
    background-color: #ADD8E6 !important;
    color: #000000 !important;
    border: 1px solid #8FC1D9 !important;
    font-weight: 600 !important;
}
.av-btn button:hover, button.av-btn:hover {
    background-color: #9ECEE6 !important;
}

/* Secondary (Back) buttons, slightly lighter */
.av-btn-secondary, .av-btn-secondary button {
    background-color: #E6F4FA !important;
    color: #000000 !important;
    border: 1px solid #BFE3F0 !important;
}

label, .label-wrap span, p, h1, h2, h3, .prose {
    color: #000000;
}
"""


# ----------------------------------------------------------------------------
# Screen navigation helpers
# ----------------------------------------------------------------------------
def show_face_screen():
    return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)


def show_object_screen():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)


def show_home_screen():
    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)


# ----------------------------------------------------------------------------
# Per-frame pipeline wrappers (extra safety net around the core modules so
# a single bad frame / model exception never crashes the live stream).
# ----------------------------------------------------------------------------
def face_pipeline(frame, frame_count, face_cache):
    try:
        return face_analysis.process_frame(frame, frame_count, face_cache)
    except Exception:
        logger.exception("Unhandled error in face analysis pipeline")
        return message_frame("Model loading failed."), frame_count, face_cache


def object_pipeline(frame, engine, prompts, confidence, frame_count):
    try:
        return object_detection.process_frame(frame, engine, prompts, confidence, frame_count)
    except Exception:
        logger.exception("Unhandled error in object detection pipeline")
        return message_frame("Model loading failed."), frame_count


# ----------------------------------------------------------------------------
# Build the UI
# ----------------------------------------------------------------------------
with gr.Blocks(
    title=config.APP_TITLE,
    css=CUSTOM_CSS,
    theme=gr.themes.Base(primary_hue="sky", neutral_hue="slate"),
) as demo:

    # ---------------- HOME SCREEN ----------------
    with gr.Column(visible=True) as home_col:
        gr.Markdown(f"<h1 class='av-title'>{config.APP_TITLE}</h1>")
        gr.Markdown(f"<p class='av-subtitle'>{config.APP_SUBTITLE}</p>")
        with gr.Row():
            face_btn = gr.Button("Face Analysis", elem_classes="av-btn")
            object_btn = gr.Button("Object Detection", elem_classes="av-btn")
        gr.Markdown(
            "<p class='av-note'>Choose a mode above, then allow camera access "
            "in your browser. Analysis starts immediately and updates live.</p>"
        )

    # ---------------- FACE ANALYSIS SCREEN ----------------
    with gr.Column(visible=False) as face_col:
        gr.Markdown("<h2 class='av-title'>Face Analysis</h2>")
        gr.Markdown(
            "<p class='av-subtitle'>Live face detection, emotion recognition, "
            "and gender prediction.</p>"
        )
        with gr.Row():
            face_input = gr.Image(
                sources=["webcam"], streaming=True, type="numpy", label="Webcam Input"
            )
            face_output = gr.Image(type="numpy", label="Live Analysis")

        face_frame_count = gr.State(0)
        face_cache_state = gr.State({})
        face_back_btn = gr.Button("Back to Home", elem_classes="av-btn-secondary")

        face_input.stream(
            fn=face_pipeline,
            inputs=[face_input, face_frame_count, face_cache_state],
            outputs=[face_output, face_frame_count, face_cache_state],
            stream_every=config.STREAM_EVERY_SECONDS,
            time_limit=None,
        )

    # ---------------- OBJECT DETECTION SCREEN ----------------
    with gr.Column(visible=False) as object_col:
        gr.Markdown("<h2 class='av-title'>Object Detection</h2>")
        gr.Markdown(
            "<p class='av-subtitle'>Live, open-vocabulary object detection — "
            "recognizes far more than the standard 80 COCO classes.</p>"
        )
        with gr.Row():
            engine_dropdown = gr.Dropdown(
                choices=config.DETECTION_ENGINES,
                value=config.DEFAULT_DETECTION_ENGINE,
                label="Detection Engine",
                info="YOLO-World = fastest, real-time. Grounding DINO / Florence-2 = slower, higher accuracy.",
            )
            confidence_slider = gr.Slider(
                minimum=0.05, maximum=0.9, step=0.05,
                value=config.DEFAULT_CONFIDENCE_THRESHOLD,
                label="Confidence Threshold",
            )
        prompt_box = gr.Textbox(
            label="Objects to Detect (comma-separated — leave as-is for the default list)",
            value=", ".join(config.OPEN_VOCAB_OBJECTS),
            lines=2,
        )
        with gr.Row():
            object_input = gr.Image(
                sources=["webcam"], streaming=True, type="numpy", label="Webcam Input"
            )
            object_output = gr.Image(type="numpy", label="Live Detections")

        object_frame_count = gr.State(0)
        object_back_btn = gr.Button("Back to Home", elem_classes="av-btn-secondary")

        object_input.stream(
            fn=object_pipeline,
            inputs=[object_input, engine_dropdown, prompt_box, confidence_slider, object_frame_count],
            outputs=[object_output, object_frame_count],
            stream_every=config.STREAM_EVERY_SECONDS,
            time_limit=None,
        )

    # ---------------- NAVIGATION WIRING ----------------
    face_btn.click(show_face_screen, outputs=[home_col, face_col, object_col])
    object_btn.click(show_object_screen, outputs=[home_col, face_col, object_col])
    face_back_btn.click(show_home_screen, outputs=[home_col, face_col, object_col])
    object_back_btn.click(show_home_screen, outputs=[home_col, face_col, object_col])


if __name__ == "__main__":
    demo.queue(max_size=20).launch()
