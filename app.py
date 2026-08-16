"""Streamlit interface for the ChestXplain research prototype."""

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image
import streamlit as st

from src.config import CFG
from src.inference import (
    DISCLAIMER,
    generate_gradcam_overlay,
    load_checkpoint_model,
    predict_probabilities,
)


HF_MODEL_REPO = "ajinkya1807/t1-mlops-stack-model"
HF_MODEL_FILENAME = "densenet121_chestxray.pth"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_GRADCAM_CLASSES = 4
RESEARCH_WARNING = (
    "Research prototype only: do not upload patient-identifiable radiographs, "
    "restricted NIH/PhysioNet data, or other private clinical images. Use only "
    "synthetic, public, or otherwise authorized images. Predictions must not "
    "replace professional medical evaluation."
)

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

st.set_page_config(
    page_title="ChestXplain - XAI Chest X-ray Diagnosis",
    page_icon="X",
    layout="wide",
)


def resolve_checkpoint_path():
    """Use a local checkpoint when available, otherwise download the public artifact."""
    configured_path = os.getenv("MODEL_PATH")
    local_candidates = []
    if configured_path:
        local_candidates.append(Path(configured_path))
    local_candidates.append(Path(CFG.MODEL_DIR) / CFG.MODEL_FILENAME)

    for candidate in local_candidates:
        if candidate.is_file():
            return str(candidate)

    return hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=HF_MODEL_FILENAME,
        revision="efa149c1489f087432b10cac2085d9b079808caf",
    )


@st.cache_resource
def load_model():
    """Load the configured local or approved public checkpoint once per process."""
    return load_checkpoint_model(model_path=resolve_checkpoint_path())


def validate_upload(uploaded):
    """Validate encoded size and decoded dimensions before image conversion."""
    if uploaded.size > MAX_UPLOAD_BYTES:
        raise ValueError("Upload exceeds the 10 MB limit.")

    uploaded.seek(0)
    with Image.open(uploaded) as opened_image:
        width, height = opened_image.size
        if width * height > MAX_IMAGE_PIXELS:
            raise ValueError("Image exceeds the 20-million-pixel limit.")
        return opened_image.convert("RGB")


def main():
    st.title("ChestXplain")
    st.markdown("**Explainable AI for Chest X-ray Disease Classification**")
    st.markdown(
        "Upload a chest X-ray and review model probabilities with a Grad-CAM "
        "overlay for the selected predictions."
    )
    st.warning(RESEARCH_WARNING)

    st.sidebar.header("Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence threshold", 0.1, 0.9, 0.5, 0.05,
        help="Diseases with probability above this are flagged as detected.",
    )
    overlay_alpha = st.sidebar.slider(
        "Heatmap opacity", 0.1, 0.8, 0.4, 0.05,
        help="How strongly the Grad-CAM overlay is blended.",
    )
    top_k = st.sidebar.slider(
        "Show top-K diseases", 1, CFG.NUM_CLASSES, 5,
        help="Number of diseases to display.",
    )

    uploaded = st.file_uploader(
        "Upload a chest X-ray (PNG, JPG)",
        type=["png", "jpg", "jpeg"],
        help="Maximum 10 MB and 20 million pixels. Do not upload private or identifiable images.",
    )

    if uploaded is None:
        st.info("Upload a chest X-ray image to get started.")
        st.markdown(
            "**Supported conditions:** " + ", ".join(CFG.DISEASE_LABELS)
        )
        st.warning(DISCLAIMER)
        return

    try:
        image = validate_upload(uploaded)
        model = load_model()
        probabilities = predict_probabilities(model, image)
    except (AttributeError, ImportError, OSError, RuntimeError, ValueError, FileNotFoundError) as exc:
        st.error(f"The model or upload is not ready: {exc}")
        st.info(
            "The app uses the approved public checkpoint when no local model is available."
        )
        return

    image_np = np.asarray(image, dtype=np.uint8)
    sorted_indices = np.argsort(probabilities)[::-1]

    col_orig, col_info = st.columns([1, 1])
    with col_orig:
        st.subheader("Uploaded X-ray")
        st.image(image, use_container_width=True)

    with col_info:
        st.subheader("Predictions")
        for idx in sorted_indices[:top_k]:
            probability = float(probabilities[idx])
            status = "Detected" if probability >= confidence_threshold else "Below threshold"
            st.markdown(f"**{CFG.DISEASE_LABELS[idx]}**: {probability:.1%} ({status})")
            st.progress(probability)

        detected = int(np.sum(probabilities >= confidence_threshold))
        if detected == 0:
            st.info("No diseases detected above the confidence threshold.")
        else:
            st.warning(f"{detected} condition(s) are above the confidence threshold.")

    st.markdown("---")
    st.subheader("Grad-CAM Explanations")
    st.caption(
        "Red regions receive greater model attribution; this is not a clinical "
        "explanation. Grad-CAM is limited to the top four displayed predictions "
        "to keep CPU inference responsive."
    )
    cam_indices = sorted_indices[: min(top_k, MAX_GRADCAM_CLASSES)]
    columns = st.columns(len(cam_indices))
    for rank, idx in enumerate(cam_indices):
        overlay = generate_gradcam_overlay(
            model, image, int(idx), alpha=overlay_alpha
        )
        with columns[rank % len(columns)]:
            st.markdown(f"**{CFG.DISEASE_LABELS[idx]}** ({probabilities[idx]:.1%})")
            st.image(overlay, use_container_width=True)

    with st.expander("All disease probabilities"):
        for idx in sorted_indices:
            st.text(f"{CFG.DISEASE_LABELS[idx]:25s}  {probabilities[idx]:.4f}")

    st.warning(RESEARCH_WARNING)
    st.warning(DISCLAIMER)
    st.caption("Built with PyTorch, FastAPI, Streamlit, and Grad-CAM.")


if __name__ == "__main__":
    main()
