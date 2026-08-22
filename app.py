"""Gradio demo for ChestXplain — research use only, not for clinical diagnosis."""

import os
from pathlib import Path

import gradio as gr
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

from src.config import CFG
from src.inference import (
    DISCLAIMER,
    generate_gradcam_overlay,
    load_checkpoint_model,
    predict_probabilities,
)

HF_MODEL_REPO = "ajinkya1807/t1-mlops-stack-model"
HF_MODEL_FILENAME = "densenet121_chestxray.pth"
HF_MODEL_REVISION = "efa149c1489f087432b10cac2085d9b079808caf"
MAX_IMAGE_PIXELS = 20_000_000

RESEARCH_WARNING = (
    "Research prototype only. Do not upload patient-identifiable radiographs, "
    "restricted NIH/PhysioNet data, or other private clinical images. "
    "Predictions must not replace professional medical evaluation."
)

_model = None


def _resolve_checkpoint():
    configured = os.getenv("MODEL_PATH")
    if configured and Path(configured).is_file():
        return configured
    local = Path(CFG.MODEL_DIR) / CFG.MODEL_FILENAME
    if local.is_file():
        return str(local)
    return hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=HF_MODEL_FILENAME,
        revision=HF_MODEL_REVISION,
    )


def _get_model():
    global _model
    if _model is None:
        _model = load_checkpoint_model(model_path=_resolve_checkpoint())
    return _model


def predict(image):
    if image is None:
        return None, []

    pil_image = Image.fromarray(image).convert("RGB") if isinstance(image, np.ndarray) else image.convert("RGB")

    if pil_image.width * pil_image.height > MAX_IMAGE_PIXELS:
        raise gr.Error(f"Image too large (max {MAX_IMAGE_PIXELS:,} pixels). Please resize and re-upload.")

    model = _get_model()
    probabilities = predict_probabilities(model, pil_image)

    sorted_indices = np.argsort(probabilities)[::-1]
    table = [
        [CFG.DISEASE_LABELS[i], round(float(probabilities[i]), 4)]
        for i in sorted_indices
    ]

    top_idx = int(sorted_indices[0])
    overlay = generate_gradcam_overlay(model, pil_image, top_idx, alpha=0.4)
    overlay_pil = Image.fromarray(np.asarray(overlay, dtype=np.uint8))

    return overlay_pil, table


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(
        type="numpy",
        label="Upload Chest X-Ray (PNG or JPG, max 10 MB)",
    ),
    outputs=[
        gr.Image(type="pil", label="Grad-CAM — top prediction"),
        gr.Dataframe(
            headers=["Pathology", "Probability"],
            datatype=["str", "number"],
            label="All 14 Predictions (sorted by confidence)",
        ),
    ],
    title="ChestXplain",
    description=(
        "DenseNet121 trained on NIH ChestX-ray14 — mean AUC 0.769 across 14 thoracic pathologies. "
        "Upload a chest X-ray to get per-pathology probabilities and a Grad-CAM explanation "
        "highlighting the regions the model weighted most for the top prediction.\n\n"
        f"⚠️ {RESEARCH_WARNING}\n\n"
        f"ℹ️ {DISCLAIMER}"
    ),
    allow_flagging="never",
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()
