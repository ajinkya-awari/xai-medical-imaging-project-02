"""Unified interface for three XAI methods: Grad-CAM, SHAP, and Integrated Gradients."""

import numpy as np
import torch
import torch.nn as nn
import cv2

from src.config import CFG
from src.gradcam import GradCAM, apply_gradcam_overlay

# captum and shap are imported lazily inside SHAPExplainer / IGExplainer __init__
# so that importing GradCAMExplainer alone does not trigger captum CUDA kernel loading,
# which would corrupt the CUDA context before GradCAM's backward pass runs.


def normalize_heatmap(attr_map):
    """Min-max normalize attribution map to [0, 1] range."""
    attr_map = np.asarray(attr_map, dtype=np.float32)
    min_val = attr_map.min()
    max_val = attr_map.max()

    if max_val - min_val < 1e-8:
        return np.zeros_like(attr_map, dtype=np.float32)

    normalized = (attr_map - min_val) / (max_val - min_val + 1e-8)
    return np.clip(normalized, 0, 1).astype(np.float32)


class GradCAMExplainer:
    """Gradient-weighted Class Activation Mapping."""

    def __init__(self, model):
        self.model = model
        self.gradcam = GradCAM(self.model)
        self.overlay_alpha = 0.5

    def explain(self, image_tensor, class_idx):
        """Generate Grad-CAM heatmap and overlay."""
        class_idx = int(class_idx)
        heatmap = self.gradcam.generate(image_tensor, class_idx)
        heatmap = cv2.resize(heatmap, (224, 224))
        assert heatmap.shape == (224, 224)
        heatmap_normalized = normalize_heatmap(heatmap)
        image_np = image_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        overlay_rgb = apply_gradcam_overlay(image_np, heatmap_normalized, alpha=self.overlay_alpha)
        assert overlay_rgb.shape == (224, 224, 3)
        assert overlay_rgb.dtype == np.uint8
        return heatmap_normalized, overlay_rgb

    def cleanup(self):
        """Remove backward hooks to prevent memory leaks."""
        self.gradcam.remove_hooks()


class SHAPExplainer:
    """SHAP Deep Explainer with GradientExplainer fallback."""

    def __init__(self, model, background):
        import shap as _shap  # lazy: do not load shap CUDA kernels until this class is used
        self.model = model
        self.background = background
        self.overlay_alpha = 0.5
        device = next(model.parameters()).device
        self.background = background.to(device)

        try:
            self.explainer = _shap.DeepExplainer(self.model, self.background)
        except (RuntimeError, Exception):
            self.explainer = _shap.GradientExplainer(self.model, [self.background])

    def explain(self, image_tensor, class_idx):
        """Generate SHAP attribution and overlay."""
        class_idx = int(class_idx)
        shap_values = self.explainer.shap_values(image_tensor, check_additivity=False)

        if isinstance(shap_values, list):
            assert len(shap_values) == CFG.NUM_CLASSES
            class_values = shap_values[class_idx]
        else:
            values = np.asarray(shap_values)
            assert values.ndim == 5 and values.shape[-1] == CFG.NUM_CLASSES
            class_values = values[..., int(class_idx)]

        assert np.asarray(class_values).shape == (1, 3, 224, 224)
        attr = np.abs(np.asarray(class_values)[0]).mean(axis=0)
        assert attr.shape == (224, 224)
        heatmap_normalized = normalize_heatmap(attr)
        image_np = image_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        overlay_rgb = apply_gradcam_overlay(image_np, heatmap_normalized, alpha=self.overlay_alpha)
        assert overlay_rgb.shape == (224, 224, 3)
        assert overlay_rgb.dtype == np.uint8
        return heatmap_normalized, overlay_rgb


class IGExplainer:
    """Integrated Gradients via Captum."""

    def __init__(self, model, n_steps=50):
        from captum.attr import IntegratedGradients  # lazy: load captum only when IG is used
        self.model = model
        self.n_steps = n_steps
        self.overlay_alpha = 0.5
        self.ig = IntegratedGradients(self.model)

    def explain(self, image_tensor, class_idx):
        """Generate Integrated Gradients attribution and overlay."""
        target = int(class_idx)
        baseline = torch.zeros_like(image_tensor)

        attrs = self.ig.attribute(
            image_tensor,
            baselines=baseline,
            target=target,
            n_steps=self.n_steps,
            return_convergence_delta=False
        )

        assert attrs.shape == (1, 3, 224, 224)
        attr = attrs.detach().abs().mean(1)[0].cpu().numpy()
        assert attr.shape == (224, 224)
        heatmap_normalized = normalize_heatmap(attr)
        image_np = image_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        overlay_rgb = apply_gradcam_overlay(image_np, heatmap_normalized, alpha=self.overlay_alpha)
        assert overlay_rgb.shape == (224, 224, 3)
        assert overlay_rgb.dtype == np.uint8
        return heatmap_normalized, overlay_rgb
