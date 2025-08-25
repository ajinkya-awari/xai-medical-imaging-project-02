
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
from src.config import CFG
from src.model import ChestXrayModel
from src.dataset import build_loaders


def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_loader, _ = build_loaders()

    model = ChestXrayModel(num_classes=CFG.NUM_CLASSES, pretrained=False).to(device)
    ckpt  = torch.load(os.path.join(CFG.MODEL_DIR, CFG.MODEL_FILENAME), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded checkpoint: epoch {ckpt['epoch']}, val AUC {ckpt['val_auc']:.4f}")

    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="  Evaluating"):
            all_preds.append(torch.sigmoid(model(images.to(device))).cpu().numpy())
            all_labels.append(labels.numpy())

    labels = np.vstack(all_labels)
    preds  = np.vstack(all_preds)
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    aucs, valid = {}, []
    for i, disease in enumerate(CFG.DISEASE_LABELS):
        if len(np.unique(labels[:, i])) > 1:
            auc = roc_auc_score(labels[:, i], preds[:, i])
            aucs[disease] = round(float(auc), 4)
            valid.append(i)
        else:
            aucs[disease] = None

    mean_auc = float(np.mean([v for v in aucs.values() if v is not None]))
    print("\n  Per-class AUC:")
    for disease, auc in aucs.items():
        bar = "█" * int((auc or 0) * 30)
        print(f"  {disease:22s}  {str(auc):>6}  {bar}")
    print(f"  {'Mean AUC':22s}  {mean_auc:.4f}")

    with open(os.path.join(CFG.OUTPUT_DIR, "test_results.json"), "w") as f:
        json.dump({"per_class_auc": aucs, "mean_auc": mean_auc,
                   "epoch": ckpt["epoch"], "val_auc": ckpt["val_auc"]}, f, indent=2)

    # AUC bar chart
    vd = [d for d, v in aucs.items() if v is not None]
    va = [aucs[d] for d in vd]
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(vd, va, color=["#e74c3c" if a < 0.7 else "#2ecc71" for a in va], edgecolor="white")
    ax.axvline(mean_auc, color="#3498db", linestyle="--", lw=2, label=f"Mean AUC = {mean_auc:.3f}")
    ax.axvline(0.5, color="grey", linestyle=":", lw=1, label="Random (0.5)")
    for bar, auc in zip(bars, va):
        ax.text(auc + 0.005, bar.get_y() + bar.get_height()/2, f"{auc:.3f}", va="center", fontsize=9)
    ax.set_xlabel("AUC-ROC", fontsize=12)
    ax.set_title("DenseNet121 — NIH ChestX-ray14 Test AUC", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.05); ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(CFG.OUTPUT_DIR, "auc_barplot.png"), dpi=150); plt.close()

    # ROC curves
    ncols = 4; nrows = (len(valid) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3.5))
    axes = axes.flatten()
    for pi, ci in enumerate(valid):
        fpr, tpr, _ = roc_curve(labels[:, ci], preds[:, ci])
        axes[pi].plot(fpr, tpr, color="#2980b9", lw=2)
        axes[pi].plot([0,1],[0,1], "k--", lw=1)
        axes[pi].set_title(f"{CFG.DISEASE_LABELS[ci]}\nAUC={aucs[CFG.DISEASE_LABELS[ci]]:.3f}", fontsize=9)
        axes[pi].set_xlabel("FPR", fontsize=8); axes[pi].set_ylabel("TPR", fontsize=8)
    for ax in axes[len(valid):]: ax.set_visible(False)
    plt.suptitle("ROC Curves — NIH ChestX-ray14", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.OUTPUT_DIR, "roc_curves.png"), dpi=150, bbox_inches="tight"); plt.close()

    print("  Saved: auc_barplot.png, roc_curves.png, test_results.json")
    return mean_auc
