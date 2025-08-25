
import os, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from src.config import CFG
from src.model import ChestXrayModel
from src.gradcam import GradCAM, apply_gradcam_overlay
from src.dataset import build_loaders, get_transforms


def generate_samples(n_samples=6):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, _, test_records = build_loaders()
    model = ChestXrayModel(num_classes=CFG.NUM_CLASSES, pretrained=False).to(device)
    ckpt  = torch.load(os.path.join(CFG.MODEL_DIR, CFG.MODEL_FILENAME), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    transform = get_transforms("val")
    cam       = GradCAM(model)
    samples   = random.sample(test_records, min(n_samples, len(test_records)))

    fig, axes = plt.subplots(len(samples), 4, figsize=(14, len(samples) * 3.5))
    if len(samples) == 1: axes = [axes]

    for row, (img_path, gt_labels) in enumerate(samples):
        image  = Image.open(img_path).convert("RGB")
        img_np = np.array(image.resize((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE)))
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            probs = torch.sigmoid(model(tensor.to(device))).cpu().numpy()[0]
        top3 = np.argsort(probs)[::-1][:3]
        gt   = ", ".join([CFG.DISEASE_LABELS[i] for i in range(CFG.NUM_CLASSES) if gt_labels[i]])
        axes[row][0].imshow(img_np, cmap="gray")
        axes[row][0].set_title(f"GT: {gt or 'No Finding'}", fontsize=7, wrap=True)
        axes[row][0].axis("off")
        for col, idx in enumerate(top3, start=1):
            overlay = apply_gradcam_overlay(img_np, cam.generate(tensor, class_idx=int(idx)), alpha=0.4)
            axes[row][col].imshow(overlay)
            axes[row][col].set_title(f"{CFG.DISEASE_LABELS[idx]}\n{probs[idx]:.1%}", fontsize=8)
            axes[row][col].axis("off")

    cam.cleanup()
    plt.suptitle("Grad-CAM — DenseNet121 on NIH ChestX-ray14", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = os.path.join(CFG.OUTPUT_DIR, "gradcam_samples.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")
