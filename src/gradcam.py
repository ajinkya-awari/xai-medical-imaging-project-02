
import numpy as np
import cv2
import torch


class GradCAM:
    def __init__(self, model):
        self.model  = model
        self.grads  = None
        self.acts   = None
        self._hooks = []
        target = model.features.denseblock4
        self._hooks.append(target.register_forward_hook(lambda m, i, o: setattr(self, "acts", o)))
        self._hooks.append(target.register_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0])))

    def generate(self, input_tensor, class_idx):
        self.model.eval()
        device = next(self.model.parameters()).device
        x = input_tensor.to(device).requires_grad_(True)
        logits = self.model(x)
        self.model.zero_grad()
        logits[0, class_idx].backward()
        grads   = self.grads[0].cpu().detach().numpy()
        acts    = self.acts[0].cpu().detach().numpy()
        weights = grads.mean(axis=(1, 2))
        cam     = sum(w * acts[k] for k, w in enumerate(weights))
        cam     = np.maximum(cam, 0)
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    def cleanup(self):
        for h in self._hooks:
            h.remove()


def apply_gradcam_overlay(image_np, heatmap, alpha=0.4):
    h, w  = image_np.shape[:2]
    heat  = cv2.resize(heatmap, (w, h))
    heat  = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
    heat  = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    out   = (1 - alpha) * image_np.astype(np.float32) + alpha * heat.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)
