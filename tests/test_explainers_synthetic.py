import numpy as np
import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from src.config import CFG
from src.explainers import GradCAMExplainer, SHAPExplainer, IGExplainer, normalize_heatmap


def test_normalize_heatmap_bounded():
    heatmap = np.array([[0.1, 0.5], [0.3, 0.9]], dtype=np.float32)
    normalized = normalize_heatmap(heatmap)
    assert normalized.shape == heatmap.shape
    assert 0.0 <= normalized.min() and normalized.max() <= 1.0


def test_normalize_heatmap_constant():
    heatmap = np.ones((224, 224), dtype=np.float32) * 0.5
    normalized = normalize_heatmap(heatmap)
    assert normalized.shape == (224, 224)
    assert np.allclose(normalized, 0.0)


def test_gradcam_explainer_shape():
    with patch("src.explainers.GradCAM") as mock_class:
        mock_grad = Mock()
        mock_class.return_value = mock_grad
        mock_grad.generate.return_value = np.random.rand(224, 224).astype(np.float32)
        
        fake_model = Mock()
        explainer = GradCAMExplainer(fake_model)
        
        with patch("src.explainers.apply_gradcam_overlay") as mock_overlay:
            mock_overlay.return_value = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            image = torch.randn(1, 3, 224, 224)
            heatmap, overlay = explainer.explain(image, 0)
            
            assert heatmap.shape == (224, 224)
            assert overlay.shape == (224, 224, 3)
            assert overlay.dtype == np.uint8


def test_ig_explainer_shape():
    with patch("captum.attr.IntegratedGradients") as mock_class:
        mock_ig = Mock()
        mock_class.return_value = mock_ig
        mock_ig.attribute.return_value = torch.randn(1, 3, 224, 224)

        fake_model = Mock()
        fake_model.parameters.return_value = [torch.randn(1)]
        explainer = IGExplainer(fake_model)
        
        with patch("src.explainers.apply_gradcam_overlay") as mock_overlay:
            mock_overlay.return_value = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            image = torch.randn(1, 3, 224, 224)
            heatmap, overlay = explainer.explain(image, 0)
            
            assert heatmap.shape == (224, 224)
            assert overlay.shape == (224, 224, 3)


def test_all_class_indices():
    with patch("src.explainers.GradCAM") as mock_class:
        mock_grad = Mock()
        mock_class.return_value = mock_grad
        mock_grad.generate.return_value = np.random.rand(224, 224).astype(np.float32)
        
        fake_model = Mock()
        explainer = GradCAMExplainer(fake_model)
        
        with patch("src.explainers.apply_gradcam_overlay") as mock_overlay:
            mock_overlay.return_value = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            image = torch.randn(1, 3, 224, 224)
            
            for idx in range(0, CFG.NUM_CLASSES, 3):
                heatmap, overlay = explainer.explain(image, idx)
                assert heatmap.shape == (224, 224)
                assert overlay.shape == (224, 224, 3)
