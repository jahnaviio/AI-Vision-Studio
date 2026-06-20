---
title: AI Vision Studio
emoji: 🎥
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 4.36.0
app_file: app.py
pinned: false
license: mit
---

# AI Vision Studio

**Real-Time Face Analysis and Object Detection** — a live webcam computer
vision platform built with OpenCV, DeepFace, YOLO-World, Grounding DINO,
Florence-2, and Gradio.

This is a **live-stream** application, not an image-upload tool. Open the
app, allow camera access, and analysis begins immediately and updates
continuously.

## Features

- **Face Detection** — OpenCV-based, draws a bounding box around every
  face in the frame, continuously.
- **Emotion Recognition** — DeepFace classifies each face into Happy,
  Sad, Angry, Fear, Surprise, Neutral, or Disgust, with a confidence
  percentage, e.g. `Happy (92%)`.
- **Gender Prediction** — DeepFace predicts `Woman (96%)` / `Man (94%)`,
  or shows `Prediction Uncertain` when confidence is too low.
- **Open-Vocabulary Object Detection** — recognizes far more object
  categories than the standard 80-class COCO set (pens, chargers,
  earbuds, remotes, backpacks, and more), via a choice of three engines:
  - **YOLO-World** (default) — fast, genuinely real-time
  - **Grounding DINO** — slower, higher-accuracy zero-shot grounding
  - **Florence-2** — Microsoft's unified vision-language model
- **Live Webcam Processing** — continuous frame-by-frame analysis with
  on-screen labels and confidence scores, no upload step.

## Tech Stack

| Layer            | Technology                                   |
|-------------------|----------------------------------------------|
| Face analysis      | OpenCV, DeepFace                            |
| Object detection    | YOLO-World (Ultralytics), Grounding DINO, Florence-2 (HF Transformers) |
| Interface           | Gradio (Blocks + live webcam streaming)     |
| Language            | Python 3.10+                                |
| Deployment          | Hugging Face Spaces, GitHub                 |

## Project Structure

```
ai-vision-studio/
├── app.py                   # Gradio UI + screen navigation + stream wiring
├── config.py                 # Centralized settings (models, colors, thresholds)
├── requirements.txt           # Python dependencies
├── packages.txt                # apt packages for Hugging Face Spaces
├── README.md
├── LICENSE
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── face_analysis.py        # OpenCV detection + DeepFace emotion/gender
│   ├── object_detection.py      # YOLO-World / Grounding DINO / Florence-2
│   └── utils.py                 # Drawing helpers, error frames
├── tests/
│   └── test_smoke.py             # Lightweight import/utility tests
└── assets/                        # Screenshots / static assets
```

## How It Works

### Home Screen
Two buttons — **Face Analysis** and **Object Detection** — switch
between modes. No upload controls anywhere in the app; both modes use
`gr.Image(sources=["webcam"], streaming=True)` wired to `.stream()`,
so frames are pulled from the browser's camera continuously and run
through the pipeline in near real time.

### Face Analysis Mode
1. The browser requests webcam permission.
2. Every frame: OpenCV's Haar Cascade detects face bounding boxes (cheap,
   runs every frame for a smooth box).
3. Every few frames (`config.EMOTION_GENDER_FRAME_SKIP`): each face crop
   is passed to `DeepFace.analyze(actions=["emotion", "gender"])`. The
   result is cached and reused on in-between frames so the label never
   flickers — it just refreshes periodically, keeping the stream smooth
   on CPU-only hardware.
4. Labels are drawn directly above each face: `Happy (92%)` and
   `Woman (96%)` (or `Prediction Uncertain` below the confidence floor).

### Object Detection Mode
1. The browser requests webcam permission.
2. Every frame is sent to the selected detection engine along with a
   comma-separated open-vocabulary prompt list (editable in the UI,
   defaults to `config.OPEN_VOCAB_OBJECTS`).
3. Detected boxes are drawn with `Label (Score%)` tags, e.g.
   `Phone (95%)`, `Book (91%)`, `Remote (93%)`.
4. Switch engines from the dropdown:
   - **YOLO-World** — recommended default, smooth real-time performance.
   - **Grounding DINO** / **Florence-2** — heavier transformer models,
     more accurate at grounding unusual phrases but noticeably slower
     per frame on CPU. Florence-2's plain detection task does not emit
     confidence scores, so its tags show the label only.

### Error Handling
- No camera / permission denied → frame shows **`Camera access required.`**
- Any model fails to load or errors mid-stream → frame shows
  **`Model loading failed.`** (with a short diagnostic sub-message), and
  the stream keeps running so the user can retry without reloading the
  page.

## Getting Started Locally

```bash
git clone https://github.com/<your-username>/ai-vision-studio.git
cd ai-vision-studio

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python app.py
```

Then open the local URL Gradio prints (typically `http://127.0.0.1:7860`)
and allow camera access when prompted.

> **First-run note:** YOLO-World weights, DeepFace's emotion/gender
> models, and (if selected) Grounding DINO / Florence-2 weights are
> downloaded automatically on first use and cached locally — the very
> first prediction in each mode may take longer while this happens.

## Deploying to Hugging Face Spaces

1. Create a new Space → SDK: **Gradio**.
2. Push this repository's contents to the Space's git remote (or
   upload via the Spaces web UI). The YAML block at the top of this
   README configures the Space automatically (`app_file: app.py`).
3. `packages.txt` installs the required system libraries
   (`libgl1`, `libglib2.0-0`, `ffmpeg`) before Python dependencies.
4. **Hardware:** the CPU Basic tier runs Face Analysis and
   YOLO-World Object Detection at usable real-time speeds. For smoother
   performance with Grounding DINO or Florence-2, an upgraded CPU or a
   GPU Space is recommended.
5. Webcam access requires the Space to be served over **HTTPS**, which
   Hugging Face Spaces provides by default.

## Performance Notes & Limitations

- This is a CPU-friendly demo by default. Emotion/gender analysis is
  intentionally throttled (`EMOTION_GENDER_FRAME_SKIP`) to keep the
  video feed smooth instead of analyzing every single frame.
- YOLO-World is the only engine tuned for true real-time use;
  Grounding DINO and Florence-2 trade frame rate for grounding accuracy
  and are best treated as an "accuracy mode" rather than the default.
- Gradio's webcam streaming processes one frame at a time per session;
  this project targets a single active user per Space instance, in
  keeping with its scope as a student/demo project rather than a
  multi-tenant production service.
- Lighting, camera angle, and occlusion affect DeepFace's emotion and
  gender accuracy, as with any face-analysis model.

## Troubleshooting

| Message | Likely cause | Fix |
|---|---|---|
| `Camera access required.` | Browser permission denied / no camera found | Allow camera access in your browser's site settings and reload |
| `Model loading failed.` | Missing dependency, failed weight download, or no internet on first run | Check the sub-message shown in the frame, verify `requirements.txt` installed cleanly, and confirm outbound internet access for model downloads |
| Slow first frame in Object Detection | Model weights downloading for the first time | Wait for the one-time download to finish; subsequent runs use the local cache |

## License

Released under the [MIT License](LICENSE).
