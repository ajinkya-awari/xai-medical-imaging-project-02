"""Benchmark all three XAI methods against NIH ChestX-ray14 BBox annotations using IoU."""

import json
from pathlib import Path

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


def compute_iou(attr_map, x, y, w, h, orig_width, orig_height, pct=90):
    """Compute IoU between attribution map and bounding box."""
    sx = CFG.IMAGE_SIZE / float(orig_width)
    sy = CFG.IMAGE_SIZE / float(orig_height)
    bx, by = int(float(x) * sx), int(float(y) * sy)
    bw, bh = int(float(w) * sx), int(float(h) * sy)

    assert 0 <= bx < CFG.IMAGE_SIZE and 0 <= by < CFG.IMAGE_SIZE, f"Bbox out of bounds: {bx}, {by}"
    assert bw > 0 and bh > 0, f"Bbox dimensions invalid: {bw}, {bh}"

    bbox_mask = np.zeros((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE), dtype=bool)
    bbox_mask[by : by + bh, bx : bx + bw] = True

    threshold = np.percentile(attr_map, pct)
    attr_mask = attr_map > threshold

    inter = (attr_mask & bbox_mask).sum()
    union = (attr_mask | bbox_mask).sum()
    return 0.0 if union == 0 else float(inter) / float(union)


def run_benchmark(model, device, explainers, bbox_df, data_dir, method_name):
    """Run benchmark for one XAI method over all BBox images."""
    results = {
        "method": method_name,
        "per_image": [],
        "per_pathology": {},
        "mean_iou": None,
    }

    for pathology in CFG.BBOX_PATHOLOGIES:
        results["per_pathology"][pathology] = []

    for idx, row in tqdm(bbox_df.iterrows(), total=len(bbox_df), desc=f"Benchmarking {method_name}"):
        image_idx = row["image_index"]
        pathology = row["finding_label"]
        class_idx = CFG.DISEASE_LABELS.index(pathology)

        image_path = data_dir / f"{image_idx}.png"
        if not image_path.is_file():
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = preprocess_image(image).to(device)

            explainer = explainers[method_name]
            heatmap, _ = explainer.explain(image_tensor, class_idx)

            iou = compute_iou(
                heatmap,
                row["x"],
                row["y"],
                row["w"],
                row["h"],
                row["orig_width"],
                row["orig_height"],
                pct=90,
            )

            results["per_image"].append(
                {
                    "image_ordinal": len(results["per_image"]),
                    "pathology": pathology,
                    "iou": float(iou),
                }
            )
            results["per_pathology"][pathology].append(float(iou))
        except Exception as e:
            print(f"Error on {image_idx} ({pathology}): {e}")
            continue

    if results["per_image"]:
        all_ious = [img["iou"] for img in results["per_image"]]
        results["mean_iou"] = float(np.mean(all_ious))

        for pathology in results["per_pathology"]:
            if results["per_pathology"][pathology]:
                results["per_pathology"][pathology] = float(np.mean(results["per_pathology"][pathology]))
            else:
                results["per_pathology"][pathology] = None

    return results


def main():
    """Run full IoU benchmark for all three XAI methods."""
    if not Path(CFG.BBOX_PATH).is_file():
        raise FileNotFoundError(f"BBox CSV not found at {CFG.BBOX_PATH}")

    bbox_df = pd.read_csv(CFG.BBOX_PATH).rename(columns=BBOX_RENAME)
    print(f"Columns after rename: {bbox_df.columns.tolist()}")

    bbox_df = bbox_df[bbox_df["finding_label"].isin(CFG.BBOX_PATHOLOGIES)]
    print(f"Filtered to {len(bbox_df)} images with {len(CFG.BBOX_PATHOLOGIES)} annotated pathologies")

    model = load_checkpoint_model()
    device = next(model.parameters()).device
    print(f"Model loaded on {device}")

    explainers = {
        "Grad-CAM": GradCAMExplainer(model),
        "SHAP": SHAPExplainer(model, torch.randn(50, 3, 224, 224).to(device)),
        "Integrated Gradients": IGExplainer(model),
    }

    data_dir = Path(CFG.DATA_DIR) / "images"
    output_dir = Path(CFG.XAI_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for method in ["Grad-CAM", "SHAP", "Integrated Gradients"]:
        results = run_benchmark(model, device, explainers, bbox_df, data_dir, method)
        all_results[method] = results

        output_file = output_dir / f"iou_results_{method.lower().replace(' ', '_')}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {output_file}")

    print("\nIoU Summary:")
    for method, results in all_results.items():
        print(f"  {method}: mean IoU = {results['mean_iou']:.4f}")


if __name__ == "__main__":
    main()
