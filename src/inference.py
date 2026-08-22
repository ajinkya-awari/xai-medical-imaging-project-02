"""Shared, model-facing inference helpers for the API and Streamlit app."""

import base64
import io
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from src.config import CFG


DISCLAIMER = (
    "This is a research prototype, not a clinical diagnostic system. "
    "Predictions must not replace professional medical evaluation."
)
NORMALIZE_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
NORMALIZE_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def get_model_path(model_path=None):
    """Resolve the checkpoint from an explicit path, env var, or local config."""
    configured_path = model_path or os.getenv("MODEL_PATH")
    return Path(configured_path or Path(CFG.MODEL_DIR) / CFG.MODEL_FILENAME)


def load_checkpoint_model(model_path=None, device=None):
    """Load a trained checkpoint; never silently fall back to random weights."""
    resolved_path = get_model_path(model_path)
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found at {resolved_path}. "
            "Set MODEL_PATH or provide the approved checkpoint before prediction."
        )

    from src.model import ChestXrayModel

    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(str(resolved_path), map_location=target_device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
    if state_dict is None:
        raise ValueError("Checkpoint must contain a model_state_dict mapping.")

    model = ChestXrayModel(num_classes=CFG.NUM_CLASSES, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(target_device)
    model.eval()
    return model


def preprocess_image(image):
    """Convert a PIL image to the model's normalized, batched tensor."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    rgb = image.convert("RGB").resize((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE), Image.Resampling.BILINEAR)
    pixels = np.asarray(rgb, dtype=np.float32) / 255.0
    pixels = (pixels - NORMALIZE_MEAN) / NORMALIZE_STD
    chw = np.transpose(pixels, (2, 0, 1)).copy()
    return torch.from_numpy(chw).unsqueeze(0)


def predict_probabilities(model, image, device=None):
    """Run the model and return independent sigmoid probabilities."""
    target_device = torch.device(device or next(model.parameters()).device)
    inputs = preprocess_image(image).to(target_device)
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(inputs))[0].detach().cpu().numpy()
    if probabilities.shape != (CFG.NUM_CLASSES,):
        raise ValueError(f"Expected {CFG.NUM_CLASSES} output probabilities, got {probabilities.shape}.")
    return probabilities.astype(np.float32)


def probabilities_payload(probabilities):
    """Build the label-ordered response dict with top prediction and confidence."""
    values = np.asarray(probabilities, dtype=np.float32)
    if values.shape != (CFG.NUM_CLASSES,):
        raise ValueError(f"Expected {CFG.NUM_CLASSES} probabilities, got {values.shape}.")
    top_index = int(np.argmax(values))
    probability_map = {
        label: float(values[index]) for index, label in enumerate(CFG.DISEASE_LABELS)
    }
    return {
        "probabilities": probability_map,
        "top_prediction": {
            "label": CFG.DISEASE_LABELS[top_index],
            "probability": float(values[top_index]),
        },
        "confidence": float(values[top_index]),
    }


def png_base64(image_np):
    """Encode a uint8 RGB image as a base64 PNG string."""
    image = Image.fromarray(np.asarray(image_np, dtype=np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def generate_gradcam_overlay(model, image, class_idx, device=None, alpha=0.4):
    """Generate one class-specific RGB Grad-CAM overlay and clean up hooks."""
    input_tensor = preprocess_image(image)

    from src.gradcam import GradCAM, apply_gradcam_overlay

    image_np = np.asarray(image.convert("RGB"), dtype=np.uint8)
    cam = GradCAM(model)
    try:
        heatmap = cam.generate(input_tensor, class_idx=int(class_idx))
        return apply_gradcam_overlay(image_np, heatmap, alpha=alpha)
    finally:
        cam.cleanup()


def run_inference(model, image, device=None):
    """Return probabilities and a top-class Grad-CAM overlay for one image."""
    probabilities = predict_probabilities(model, image, device=device)
    top_index = int(np.argmax(probabilities))
    overlay = generate_gradcam_overlay(model, image, top_index, device=device)
    result = probabilities_payload(probabilities)
    result["gradcam_png_base64"] = png_base64(overlay)
    return result
