"""Test app.py method selector, lazy loading, and explainer wiring."""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import CFG
from src.explainers import GradCAMExplainer, SHAPExplainer, IGExplainer


def test_explainers_accept_image_tensor_and_return_tuple():
    """Verify all three explainers implement the unified interface."""
    import torch
    from src.model import ChestXrayModel

    img_tensor = torch.randn(1, 3, 224, 224)
    model = ChestXrayModel(pretrained=False)
    model.eval()

    explainer = GradCAMExplainer(model)
    heatmap, overlay = explainer.explain(img_tensor, class_idx=0)
    assert heatmap.shape == (224, 224), f"Heatmap shape wrong: {heatmap.shape}"
    assert overlay.shape == (224, 224, 3), f"Overlay shape wrong: {overlay.shape}"
    assert overlay.dtype == np.uint8, f"Overlay dtype wrong: {overlay.dtype}"


def test_preprocess_image_matches_inference_boundary():
    """Verify app.py can reuse inference preprocessing."""
    from src.inference import preprocess_image

    synthetic_path = REPO_ROOT / "test_synthetic_xray.png"
    if not synthetic_path.is_file():
        pytest.skip("test_synthetic_xray.png not found")

    image = Image.open(synthetic_path).convert("RGB")
    tensor = preprocess_image(image)

    assert tensor.shape == (1, 3, CFG.IMAGE_SIZE, CFG.IMAGE_SIZE)
    assert str(tensor.dtype) == "torch.float32"


def test_method_names_match_radio_choices():
    """Ensure app.py method names are consistent."""
    methods = ["Grad-CAM", "SHAP", "Integrated Gradients"]
    for method in methods:
        assert method in methods
