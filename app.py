"""Gradio demo for ChestXplain — research use only, not for clinical diagnosis."""

import os
from pathlib import Path

import gradio as gr
import numpy as np
import spaces
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from src.config import CFG
from src.explainers import GradCAMExplainer, SHAPExplainer, IGExplainer
from src.inference import (
    DISCLAIMER,
    load_checkpoint_model,
    predict_probabilities,
)

HF_MODEL_REPO = "ajinkya1807/t1-mlops-stack-model"
HF_MODEL_FILENAME = "densenet121_chestxray.pth"
HF_MODEL_REVISION = "efa149c1489f087432b10cac2085d9b079808caf"
HF_BG_FILENAME = "background_50.pt"
MAX_IMAGE_PIXELS = 20_000_000

RESEARCH_WARNING = (
    "Research prototype only. Do not upload patient-identifiable radiographs, "
    "restricted NIH/PhysioNet data, or other private clinical images. "
    "Predictions must not replace professional medical evaluation."
)

_model = None
_background = None
_explainers = {}  # Lazy-loaded dict: method_name -> explainer instance


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


def _load_background():
    global _background
    if _background is not None:
        return _background
    local_bg = Path(CFG.MODEL_DIR) / HF_BG_FILENAME
    if local_bg.is_file():
        _background = torch.load(local_bg, map_location="cpu", weights_only=True)
    else:
        bg_path = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=HF_BG_FILENAME,
            revision=HF_MODEL_REVISION,
        )
        _background = torch.load(bg_path, map_location="cpu", weights_only=True)
    return _background


def _get_explainer(method):
    global _explainers
    if method not in _explainers:
        model = _get_model()
        device = next(model.parameters()).device
        if method == "Grad-CAM":
            _explainers[method] = GradCAMExplainer(model)
        elif method == "SHAP":
            bg = _load_background().to(device)
            _explainers[method] = SHAPExplainer(model, bg, device=device)
        elif method == "Integrated Gradients":
            _explainers[method] = IGExplainer(model, device=device)
    return _explainers[method]


@spaces.GPU
def predict(image, method):
    if image is None:
        return None, [], ""

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
    device = next(model.parameters()).device
    from src.inference import preprocess_image
    image_tensor = preprocess_image(pil_image).to(device)

    explainer = _get_explainer(method)
    _, overlay_rgb = explainer.explain(image_tensor, top_idx)
    overlay_pil = Image.fromarray(np.asarray(overlay_rgb, dtype=np.uint8))

    method_label = f"**{method}** — top prediction"
    return overlay_pil, table, method_label


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(
            type="numpy",
            label="Upload Chest X-Ray (PNG or JPG, max 10 MB)",
        ),
        gr.Radio(
            choices=["Grad-CAM", "SHAP", "Integrated Gradients"],
            value="Grad-CAM",
            label="Explainability Method",
        ),
    ],
    outputs=[
        gr.Image(type="pil", label="Explainability Heatmap"),
        gr.Dataframe(
            headers=["Pathology", "Probability"],
            datatype=["str", "number"],
            label="All 14 Predictions (sorted by confidence)",
        ),
        gr.Markdown(label="Method Used"),
    ],
    title="ChestXplain",
    description=(
        "DenseNet121 trained on NIH ChestX-ray14 — mean AUC 0.769 across 14 thoracic pathologies. "
        "Upload a chest X-ray to get per-pathology probabilities and compare three explainability methods: "
        "Grad-CAM, SHAP, and Integrated Gradients.\n\n"
        f"⚠️ {RESEARCH_WARNING}\n\n"
        f"ℹ️ {DISCLAIMER}"
    ),
)

if __name__ == "__main__":
    demo.launch()
