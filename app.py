"""
Streamlit web app for chest X-ray diagnosis with Grad-CAM explanations.

Upload a chest X-ray image and the app will:
  1. Run DenseNet121 inference to predict disease probabilities
  2. Generate Grad-CAM heatmaps showing which regions influenced each prediction
  3. Display an interactive overlay with adjustable opacity

The model must be trained first (run_all.py or src.train) — the app
loads the checkpoint from models/densenet121_chestxray.pth.

Usage:
    streamlit run app.py
"""

import os
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

import torch
from torchvision import transforms

import streamlit as st

# need to set up paths before importing project modules
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import CFG
from src.model import ChestXrayModel
from src.gradcam import GradCAM, apply_gradcam_overlay


# ── Page config ────────────────────────────────────────────────────

st.set_page_config(
    page_title="ChestXplain — XAI Chest X-ray Diagnosis",
    page_icon="🫁",
    layout="wide",
)

# ── Cached model loading ──────────────────────────────────────────

@st.cache_resource
def load_model():
    """Load the trained model — cached so it only runs once."""
    model_path = Path(CFG.MODEL_DIR) / CFG.MODEL_FILENAME

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChestXrayModel(num_classes=CFG.NUM_CLASSES, pretrained=False)

    if model_path.exists():
        checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        st.sidebar.success(f"Model loaded (epoch {checkpoint['epoch']}, "
                          f"AUC: {checkpoint['val_auc']:.3f})")
    else:
        st.sidebar.warning("No trained model found — using random weights. "
                          "Run training first for real predictions.")

    model.to(device)
    model.eval()
    return model, device


# ── Image preprocessing ────────────────────────────────────────────

def preprocess_image(image: Image.Image):
    """Normalise and convert to tensor for model input."""
    transform = transforms.Compose([
        transforms.Resize((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return transform(image)


# ── Main app ───────────────────────────────────────────────────────

def main():
    st.title("🫁 ChestXplain")
    st.markdown("**Explainable AI for Chest X-ray Disease Classification**")
    st.markdown("Upload a chest X-ray and see what the model detects, "
                "along with Grad-CAM heatmaps showing *where* it's looking.")

    # sidebar controls
    st.sidebar.header("Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence threshold", 0.1, 0.9, 0.5, 0.05,
        help="Diseases with probability above this are flagged as 'detected'."
    )
    overlay_alpha = st.sidebar.slider(
        "Heatmap opacity", 0.1, 0.8, 0.4, 0.05,
        help="How strongly the Grad-CAM overlay is blended."
    )
    top_k = st.sidebar.slider(
        "Show top-K diseases", 1, 14, 5,
        help="Number of diseases to display Grad-CAM heatmaps for."
    )

    model, device = load_model()

    # file upload
    uploaded = st.file_uploader(
        "Upload a chest X-ray (PNG, JPG, DICOM not supported yet)",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded is not None:
        # load and display original
        image = Image.open(uploaded).convert("RGB")
        image_np = np.array(image.resize((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE)))

        col_orig, col_info = st.columns([1, 1])
        with col_orig:
            st.subheader("Uploaded X-ray")
            st.image(image, use_container_width=True)

        # inference
        input_tensor = preprocess_image(image).unsqueeze(0)

        with torch.no_grad():
            logits = model(input_tensor.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        # prediction table
        with col_info:
            st.subheader("Predictions")
            # sort by probability
            sorted_idx = np.argsort(probs)[::-1]

            for rank, idx in enumerate(sorted_idx[:top_k]):
                disease = CFG.DISEASE_LABELS[idx]
                prob = probs[idx]
                status = "🔴" if prob >= confidence_threshold else "⚪"
                bar_color = "red" if prob >= confidence_threshold else "gray"
                st.markdown(f"{status} **{disease}**: {prob:.1%}")
                st.progress(float(prob))

            # count detections
            n_detected = sum(1 for p in probs if p >= confidence_threshold)
            if n_detected == 0:
                st.info("No diseases detected above the confidence threshold.")
            else:
                st.warning(f"{n_detected} condition(s) detected above "
                          f"{confidence_threshold:.0%} threshold.")

        # Grad-CAM section
        st.markdown("---")
        st.subheader("Grad-CAM Explanations")
        st.markdown("Heatmaps show which regions of the X-ray the model focuses on "
                    "for each disease prediction. Red = high attention, Blue = low.")

        # generate heatmaps for top-K diseases
        cam = GradCAM(model)
        cols = st.columns(min(top_k, 4))

        for rank, idx in enumerate(sorted_idx[:top_k]):
            disease = CFG.DISEASE_LABELS[idx]
            prob = probs[idx]

            heatmap = cam.generate(input_tensor, class_idx=int(idx))
            overlay = apply_gradcam_overlay(image_np, heatmap, alpha=overlay_alpha)

            col_idx = rank % len(cols)
            with cols[col_idx]:
                st.markdown(f"**{disease}** ({prob:.1%})")
                st.image(overlay, use_container_width=True)

        cam.cleanup()

        # expandable details
        with st.expander("All disease probabilities"):
            for idx in sorted_idx:
                disease = CFG.DISEASE_LABELS[idx]
                prob = probs[idx]
                st.text(f"{disease:25s}  {prob:.4f}")

    else:
        # placeholder when no image uploaded
        st.info("👆 Upload a chest X-ray image to get started.")
        st.markdown("""
        **How it works:**
        1. Upload a frontal chest X-ray (PA or AP view)
        2. The DenseNet121 model predicts probabilities for 14 diseases
        3. Grad-CAM generates heatmaps showing which regions influenced each prediction
        4. Review the overlays to understand the model's reasoning

        **Supported conditions:** Atelectasis, Cardiomegaly, Effusion,
        Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation,
        Edema, Emphysema, Fibrosis, Pleural Thickening, Hernia

        ⚠️ *This is a research tool, not a clinical diagnostic system.
        Always consult a qualified radiologist.*
        """)

    # footer
    st.markdown("---")
    st.markdown(
        "<small>Built with PyTorch & Streamlit | "
        "Model: DenseNet121 pretrained on ImageNet, fine-tuned on NIH ChestX-ray14 | "
        "Explainability: Grad-CAM (Selvaraju et al., 2017)</small>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
