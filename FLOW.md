# FLOW - XAI Medical Imaging / Project 01

This describes the intended execution path and is updated only when the real code path changes. It is not a substitute for tests.

## Current path (existing application)

1. `app.py` starts Streamlit and accepts an uploaded image.
2. The app loads the current checkpoint through its existing model-download/load path.
3. The image is transformed for DenseNet121 inference.
4. `src/model.py` produces 14 raw logits.
5. Sigmoid probabilities and `src/gradcam.py` produce the prediction view/overlay.
6. Streamlit renders labels, probabilities, and the research disclaimer.

## Current implementation path (Day 5)

1. `src/train.py` reads `src.config.CFG`, initializes one W&B run, runs the existing warm-up/fine-tune loops, logs only verified scalar metrics in each loop, and finishes the run on success or failure.

## Planned path (Day 6-7, not implemented)

1. `src/inference.py` owns checkpoint resolution, `checkpoint["model_state_dict"]` loading, preprocessing, probabilities, Grad-CAM, and PNG encoding.
2. `api/main.py` loads the model once in FastAPI lifespan, validates PNG/JPEG uploads, calls the shared inference helpers, and returns the 14-label safety-aware response.
3. Docker starts `uvicorn api.main:app`; weights resolve from the public Hugging Face model artifact and cache locally.
4. `app.py` reuses the same inference boundary while retaining Streamlit as the Space entrypoint.

## Change-tracing rule

For every implementation change, update the affected arrow(s), name the exact files/functions, and add a regression test or smoke command to `TEST_CHECKLIST.md`.
