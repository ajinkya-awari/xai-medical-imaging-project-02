"""Generate comparison figures: original, Grad-CAM, SHAP, IG with BBox overlay."""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from src.config import CFG
from src.explainers import GradCAMExplainer, SHAPExplainer, IGExplainer
from src.inference import load_checkpoint_model, preprocess_image


BBOX_RENAME = {
    "Image Index": "image_index",
    "Finding Label": "finding_label",
    "Bbox [x": "x",
    "y": "y",
    "w": "w",
    "h]": "h",
    "OriginalImage[Width": "orig_width",
    "Height]": "orig_height",
    "OriginalImagePixelSpacing[x": "spacing_x",
    "y]": "spacing_y",
}

COMPARISON_PATHOLOGIES = ["Cardiomegaly", "Effusion", "Atelectasis", "Pneumothorax", "Mass"]
IMAGES_PER_PATHOLOGY = 2


def main():
    """Generate comparison figures for selected images."""
    if not Path(CFG.BBOX_PATH).is_file():
        raise FileNotFoundError(f"BBox CSV not found at {CFG.BBOX_PATH}")

    bbox_df = pd.read_csv(CFG.BBOX_PATH).rename(columns=BBOX_RENAME)
    bbox_df = bbox_df[bbox_df["finding_label"].isin(CFG.BBOX_PATHOLOGIES)]

    random.seed(42)
    selected_images = []
    for pathology in COMPARISON_PATHOLOGIES:
        pathology_images = bbox_df[bbox_df["finding_label"] == pathology]
        sampled = pathology_images.sample(n=min(IMAGES_PER_PATHOLOGY, len(pathology_images)))
        selected_images.extend(sampled.index.tolist())

    print(f"Selected {len(selected_images)} images for comparison")

    model = load_checkpoint_model()
    device = next(model.parameters()).device

    _bg_path = Path(CFG.MODEL_DIR) / "background_50.pt"
    if _bg_path.is_file():
        _bg = torch.load(str(_bg_path), map_location=device, weights_only=True)
    else:
        _bg = torch.randn(50, 3, 224, 224).to(device)

    explainers = {
        "Original": None,
        "Grad-CAM": GradCAMExplainer(model),
        "SHAP": SHAPExplainer(model, _bg),
        "Integrated Gradients": IGExplainer(model),
    }

    data_dir = Path(CFG.DATA_DIR) / "images"
    output_dir = Path(CFG.XAI_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid_axes = []
    fig, axes = plt.subplots(len(selected_images), 4, figsize=(16, 4 * len(selected_images)))
    if len(selected_images) == 1:
        axes = axes.reshape(1, -1)

    for plot_row, img_idx in enumerate(tqdm(selected_images, desc="Generating comparison figures")):
        row = bbox_df.loc[img_idx]
        pathology = row["finding_label"]
        class_idx = CFG.DISEASE_LABELS.index(pathology)

        image_path = data_dir / f"{row['image_index']}.png"
        if not image_path.is_file():
            continue

        image = Image.open(image_path).convert("RGB")
        image_array = np.asarray(image)
        image_tensor = preprocess_image(image).to(device)

        col = 0
        axes[plot_row, col].imshow(image_array, cmap="gray")
        rect = patches.Rectangle(
            (int(row["x"]), int(row["y"])),
            int(row["w"]),
            int(row["h"]),
            linewidth=2,
            edgecolor="white",
            facecolor="none",
        )
        axes[plot_row, col].add_patch(rect)
        axes[plot_row, col].set_title(f"Original\n({pathology})")
        axes[plot_row, col].axis("off")

        for col, method in enumerate(["Grad-CAM", "SHAP", "Integrated Gradients"], start=1):
            explainer = explainers[method]
            _, overlay = explainer.explain(image_tensor, class_idx)
            overlay_pil = Image.fromarray(np.asarray(overlay, dtype=np.uint8))
            axes[plot_row, col].imshow(overlay_pil)
            axes[plot_row, col].set_title(method)
            axes[plot_row, col].axis("off")

    plt.tight_layout()
    grid_path = output_dir / "xai_comparison_grid.png"
    plt.savefig(grid_path, dpi=100, bbox_inches="tight")
    print(f"Saved comparison grid to {grid_path}")
    plt.close()


if __name__ == "__main__":
    main()
