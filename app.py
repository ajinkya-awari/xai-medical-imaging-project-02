"""Streamlit interface for the ChestXplain research prototype."""

from pathlib import Path
import sys

import numpy as np
from PIL import Image
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import CFG
from src.inference import (
    DISCLAIMER,
    generate_gradcam_overlay,
    load_checkpoint_model,
    predict_probabilities,
)


st.set_page_config(
    page_title="ChestXplain - XAI Chest X-ray Diagnosis",
    page_icon="X",
    layout="wide",
)


@st.cache_resource
def load_model():
    """Load the configured local checkpoint once per Streamlit process."""
    return load_checkpoint_model()


def main():
    st.title("ChestXplain")
    st.markdown("**Explainable AI for Chest X-ray Disease Classification**")
    st.markdown(
        "Upload a chest X-ray and review model probabilities with a Grad-CAM "
        "overlay for the selected predictions."
    )

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
    )

    if uploaded is None:
        st.info("Upload a chest X-ray image to get started.")
        st.markdown(
            "**Supported conditions:** " + ", ".join(CFG.DISEASE_LABELS)
        )
        st.warning(DISCLAIMER)
        return

    try:
        image = Image.open(uploaded).convert("RGB")
        model = load_model()
        probabilities = predict_probabilities(model, image)
    except (AttributeError, ImportError, OSError, RuntimeError, ValueError, FileNotFoundError) as exc:
        st.error(f"The model is not ready: {exc}")
        st.info("Set MODEL_PATH or place the approved checkpoint under models/.")
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
    st.caption("Red regions receive greater model attribution; this is not a clinical explanation.")
    columns = st.columns(min(top_k, 4))
    for rank, idx in enumerate(sorted_indices[:top_k]):
        overlay = generate_gradcam_overlay(
            model, image, int(idx), alpha=overlay_alpha
        )
        with columns[rank % len(columns)]:
            st.markdown(f"**{CFG.DISEASE_LABELS[idx]}** ({probabilities[idx]:.1%})")
            st.image(overlay, use_container_width=True)

    with st.expander("All disease probabilities"):
        for idx in sorted_indices:
            st.text(f"{CFG.DISEASE_LABELS[idx]:25s}  {probabilities[idx]:.4f}")

    st.warning(DISCLAIMER)
    st.caption("Built with PyTorch, FastAPI, Streamlit, and Grad-CAM.")


if __name__ == "__main__":
    main()
