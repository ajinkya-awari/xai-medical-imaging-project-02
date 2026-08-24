# FLOW - XAI Medical Imaging / Project 02

This describes the intended execution path and is updated only when the real code path changes. It is not a substitute for tests.

## Current path (existing application)

1. `app.py` starts Streamlit and accepts an uploaded image.
2. The app loads the approved local checkpoint through `src.inference.load_checkpoint_model`; it does not auto-download or use random weights.
3. `src.inference.preprocess_image` transforms the image for DenseNet121 inference.
4. `src.model.py` produces 14 raw logits.
5. `src.inference.predict_probabilities` and `src.inference.generate_gradcam_overlay` produce the prediction view/overlay.
6. Streamlit renders labels, probabilities, overlays, and the research disclaimer.

## Current implementation path (Day 5)

1. `src/train.py` reads `src.config.CFG`, initializes one W&B run, runs the existing warm-up/fine-tune loops, logs only verified scalar metrics in each loop, and finishes the run on success or failure.

## Current serving path (Day 6-7)

1. `src/inference.py` owns checkpoint resolution, `checkpoint["model_state_dict"]` loading, preprocessing, probabilities, Grad-CAM, and PNG encoding.
2. `api/main.py` attempts one model load in FastAPI lifespan, validates PNG/JPEG uploads, calls the shared inference helpers, and returns the 14-label safety-aware response.
3. Docker starts `uvicorn api.main:app`; the approved checkpoint is supplied through the ignored `models/` mount or `MODEL_PATH`.
4. `app.py` reuses the same inference boundary for local Streamlit use.

## Change-tracing rule

For every implementation change, update the affected arrow(s), name the exact files/functions, and add a regression test or smoke command to the project evidence.

## Project 02 local path

1. Gradio app validates an uploaded image using the existing size and pixel limits.
2. `app.py` calls the existing cached `src.inference.load_checkpoint_model` and obtains normalized input through the existing inference preprocessing boundary.
3. The method selector chooses exactly one of `GradCAMExplainer`, `SHAPExplainer`, or `IGExplainer` from `src.explainers`.
4. The selected explainer returns a normalized `(224, 224)` heatmap and an RGB overlay; Gradio renders the result with the research-only disclaimer.
5. `src.xai_bench` independently normalizes BBox headers, filters the eight annotated labels, scales original-pixel coordinates, computes guarded IoU, and writes per-method artifacts only during an authorized run.
6. `src.xai_compare` selects the same seed-42 records for all methods and renders comparison figures only when approved input data is available.

This flow is a target for the current local implementation; each arrow must be backed by a focused test or source inspection before it is marked verified in the project evidence.
