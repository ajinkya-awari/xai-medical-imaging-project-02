<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:1a1f2e,100:0d47a1&height=220&section=header&text=XAI%20Medical%20Imaging&fontSize=56&fontColor=fff&animation=fadeIn&fontAlignY=38&desc=Can%20your%20model%20explain%20itself%3F%20Mine%20can%20now.&descAlignY=58&descAlign=50&descSize=18"/>

</div>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776ab?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![SHAP](https://img.shields.io/badge/SHAP-0.44%2B-7c3aed)](https://shap.readthedocs.io)
[![Captum](https://img.shields.io/badge/Captum-0.7%2B-22c55e)](https://captum.ai)
[![Kaggle](https://img.shields.io/badge/Kaggle-20%20kernel%20versions-20beff?logo=kaggle&logoColor=white)](https://kaggle.com)
[![Dataset](https://img.shields.io/badge/Dataset-NIH%20ChestX--ray14-f59e0b)](https://nihcc.app.box.com/v/ChestXray-NIHCC)
[![HF Space](https://img.shields.io/badge/HuggingFace-Live%20Demo-ff6b35?logo=huggingface&logoColor=white)](https://huggingface.co/spaces/ajinkya1807/xai-medical-imaging)
[![License](https://img.shields.io/badge/License-MIT-64748b)](LICENSE)

</div>

---

## What This Actually Is

I trained DenseNet121 on NIH ChestX-ray14 for 14-class chest pathology classification. Then I asked the harder question: *is the model looking at the right region?*

This repo answers that by running three explanation methods — Grad-CAM, SHAP, and Integrated Gradients — and checking each one against 861 radiologist-annotated bounding boxes using Intersection-over-Union. Not vibes. Numbers.

**Short answer:** Grad-CAM localises cardiomegaly at IoU 0.34. SHAP gets there too, differently. Nodule remains genuinely hard for all three.

---

## Benchmark Results

IoU against NIH ChestX-ray14 bounding box annotations. 90th-percentile threshold on attribution maps. Coordinates scaled from 1024×1024 → 224×224.

| Pathology | Grad-CAM | Integrated Gradients | SHAP |
|---|:---:|:---:|:---:|
| Atelectasis | 0.1047 | 0.0561 | 0.0600 |
| Cardiomegaly | **0.3373** | **0.2111** | **0.1468** |
| Effusion | 0.1757 | 0.0815 | 0.0746 |
| Infiltration | — | — | — |
| Mass | 0.0938 | 0.0501 | 0.0540 |
| Nodule | 0.0172 | 0.0094 | 0.0153 |
| Pneumonia | 0.0066 | 0.0631 | 0.0743 |
| Pneumothorax | 0.0603 | 0.0230 | 0.0227 |
| **Mean IoU** | **0.1289** | **0.0792** | **0.0704** |

861 records · 8 pathologies · 0 errors · reproducible with `random.seed(42)`

**What the numbers say:** Cardiomegaly is easiest across all methods — large, well-defined cardiac silhouette. Nodule is the hardest (IoU < 0.02) because gradient-based attribution smears across the lung field when the lesion is small and diffuse. Grad-CAM leads overall, which matches the gradient saturation literature for multi-label classifiers. SHAP and IG both underperform on Pneumonia using gradient approaches but catch different patterns — suggesting ensemble explanation could be more informative than any single method.

---

## Artifacts

| Artifact | Link |
|---|---|
| GitHub | [ajinkya-awari/xai-medical-imaging-project-02](https://github.com/ajinkya-awari/xai-medical-imaging-project-02) |
| HF Model Repo | [ajinkya1807/t1-mlops-stack-model](https://huggingface.co/ajinkya1807/t1-mlops-stack-model) |
| Live Demo (Gradio) | [ajinkya1807/xai-medical-imaging](https://huggingface.co/spaces/ajinkya1807/xai-medical-imaging) |
| Kaggle Benchmark Notebook | [02-project on Kaggle](https://www.kaggle.com/code/ajinkya1225/02-project) |

---

## The Journey: V1 to V20

This didn't work on the first try. Or the fifth. Getting three XAI methods to agree on a benchmark protocol took 20 Kaggle kernel iterations over several days. Here's what actually happened.

### Why so many versions?

The short version: SHAP's Python API is a minefield, Kaggle's T4 GPU silently breaks backward passes on certain DenseNet architectures, and getting a reproducible 861-image benchmark to run end-to-end on CPU without timing out required more patience than I expected.

```
V1–V3   Initial Grad-CAM implementation + DenseNet121 checkpoint loading
V4–V6   SHAP DeepExplainer first attempts — additivity check failures
V7–V9   Grad-CAM benchmark working ✓ (mean IoU 0.1289 confirmed)
V10–V12 Integrated Gradients benchmark working ✓ (mean IoU 0.0792 confirmed)
V13     SHAP with nsamples=200 → CPU timeout projected at 11+ hours
V14     Reduced nsamples, hit DeepExplainer's no-nsamples wall
V15     Synthetic validation gate (13/13 passing), CUDA crash on Kaggle GPU
V16     GPU completely abandoned, CUDA backward incompatible with T4+DenseNet
V17     CPU + nsamples=10, DeepExplainer doesn't accept nsamples → all 861 fail
V18     Swapped to GradientExplainer primary, hit check_additivity rejection
V19     Removed check_additivity from GradientExplainer path → list format error
V20     [image_tensor] wrapped in list to match GradientExplainer init → ✓ done
```

Every version taught me something about how SHAP actually works under the hood versus what the documentation implies.

---

## The Bugs That Cost the Most Time

### 1. SHAP DeepExplainer additivity check failures

DeepExplainer validates that SHAP values sum to the model output difference. DenseNet121 with BatchNorm violates this at tolerance 0.01 — not because the implementation is wrong, but because BatchNorm's running statistics behave differently under SHAP's repeated forward passes.

**Fix:** `shap_values(..., check_additivity=False)`. But this only applies to DeepExplainer — GradientExplainer doesn't accept this parameter at all, which cost another version.

### 2. Kaggle T4 GPU backward kernel crash

Kaggle's T4 GPU threw `cudaErrorNoKernelImageForDevice` during the backward pass through DenseNet121. This is a hardware-level CUDA kernel incompatibility — not fixable by changing PyTorch flags or model config. Lost two kernel versions learning this wasn't a code bug.

**Fix:** `enable_gpu: false` in `kernel-metadata.json`. CPU-only for the full benchmark.

### 3. GradientExplainer input format

`shap.GradientExplainer` initialised with a list `[background_tensor]` expects `shap_values([image_tensor])` — also a list. Pass a bare tensor and you get `"Expected a list of model inputs!"`. DeepExplainer is the opposite — it takes bare tensors.

**Fix:** Branching explain logic based on `self._is_gradient` flag, with separate kwargs per explainer type:
```python
if self._is_gradient:
    shap_values = self.explainer.shap_values([image_tensor], nsamples=self.nsamples)
else:
    shap_values = self.explainer.shap_values(image_tensor, check_additivity=False)
```

### 4. BBox CSV column names with literal brackets

The NIH ChestX-ray14 bounding box CSV has column names like `"Bbox [x"`, `"y"`, `"w"`, `"h]"` — literal square brackets as part of the string. Standard pandas access fails silently or raises KeyError. Had to print `df.columns.tolist()` to even see what I was dealing with.

**Fix:** Rename immediately after `read_csv`:
```python
df.columns = ["image_index", "finding_label", "x", "y", "w", "h", "orig_w", "orig_h"]
```

### 5. Background tensor device mismatch

SHAP's DeepExplainer raises a cryptic tensor device error if background and model are on different devices. PyTorch's error message points at SHAP internals, not your code.

**Fix:** `background = background.to(device)` before passing to any SHAP constructor. Added to `SHAPExplainer.__init__` explicitly.

---

## Architecture

```
NIH ChestX-ray14 (local, never uploaded)
        │
        ▼
densenet121_chestxray.pth ──► HF Hub: ajinkya1807/t1-mlops-stack-model
        │
        ▼
src/inference.py
  └── load_checkpoint_model(model_path, device)
        │
        ├──► src/explainers.py
        │        ├── GradCAMExplainer   wraps src/gradcam.py (unchanged)
        │        ├── SHAPExplainer      shap.GradientExplainer + DeepExplainer fallback
        │        └── IGExplainer        captum.IntegratedGradients, lazy import
        │
        ├──► src/xai_bench.py    → outputs/xai/iou_results_{method}.json
        ├──► src/xai_compare.py  → outputs/xai/xai_comparison_grid.png
        └──► app.py              → Gradio Space, method selector, lazy instantiation
```

All three explainers implement the same interface:
```python
heatmap_2D, overlay_rgb = explainer.explain(image_tensor, class_idx)
# heatmap_2D: (224, 224) float32 in [0, 1]
# overlay_rgb: (224, 224, 3) uint8
```

---

## IoU Evaluation Protocol

Attribution maps are not directly comparable across methods — Grad-CAM outputs [0,1] naturally, SHAP values are unbounded, IG attributions can be negative. Before any thresholding:

1. **Min-max normalise** each map: `(x - x.min()) / (x.max() - x.min() + 1e-8)`
2. **Threshold at 90th percentile**: selects the brightest 10% of each attribution map
3. **Binary mask vs BBox**: compute IoU between the mask and the scaled bounding box
4. **Scale BBox**: NIH annotations are in 1024×1024 space, model input is 224×224

```python
threshold = np.percentile(attr_map, 90)
attr_mask = attr_map > threshold
iou = intersection / union if union > 0 else 0.0
```

Only 8 of 14 pathologies have bounding box annotations in ChestX-ray14. Infiltration is excluded from mean IoU — the dataset annotations for it are incomplete.

---

## XAI Methods

| Method | Implementation | Notes |
|---|---|---|
| Grad-CAM | `src/gradcam.py` → `GradCAMExplainer` | Gradient backprop through DenseBlock4 |
| SHAP | `shap.GradientExplainer` with DeepExplainer fallback | `background_50.pt` from training split |
| Integrated Gradients | Captum, 50 steps, zero baseline | Lazy import to avoid CUDA context corruption |

All share the same `.explain()` contract. The `app.py` only instantiates the selected method at runtime — loading all three simultaneously would push an HF Space over memory limits.

---

## Setup

```bash
git clone https://github.com/ajinkya-awari/xai-medical-imaging-project-02.git
cd xai-medical-imaging-project-02

python -m venv .venv
# Windows:
.\.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate

pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Run the Gradio demo

```bash
python app.py
```

Downloads `densenet121_chestxray.pth` from HF Hub if not present locally.

### Run the IoU benchmark

Requires NIH ChestX-ray14 locally. Set paths in `src/config.py`:

```bash
python -m src.xai_bench
```

Results write to `outputs/xai/iou_results_{method}.json`.

---

## Repository Layout

```
xai-medical-imaging-project-02/
├── src/
│   ├── config.py         # CFG: paths, disease labels, XAI settings
│   ├── explainers.py     # GradCAMExplainer, SHAPExplainer, IGExplainer
│   ├── xai_bench.py      # IoU benchmark against NIH BBox annotations
│   ├── xai_compare.py    # 10-image × 3-method comparison grid
│   ├── inference.py      # load_checkpoint_model, preprocess_image
│   ├── gradcam.py        # Grad-CAM (unchanged from T1 MLOps)
│   ├── dataset.py        # DataLoader for background tensor sampling
│   └── model.py          # DenseNet121 architecture
├── api/
│   └── main.py           # FastAPI /health, /metadata, POST /predict
├── tests/
│   ├── test_explainers_synthetic.py   # 13 synthetic shape/dtype tests
│   ├── test_inference_api_contract.py
│   └── test_app_wiring.py
├── notebooks/
│   └── kaggle_xai_triple.ipynb        # Full benchmark (V20 — the one that worked)
├── outputs/xai/           # JSON results + figures (generated, gitignored)
├── app.py                 # Gradio Space
├── Dockerfile
└── requirements.txt
```

---

## References

- Selvaraju et al. 2017 — Grad-CAM ([arXiv:1610.02055](https://arxiv.org/abs/1610.02055))
- Lundberg & Lee 2017 — SHAP ([arXiv:1705.07874](https://arxiv.org/abs/1705.07874))
- Sundararajan et al. 2017 — Integrated Gradients ([arXiv:1703.01365](https://arxiv.org/abs/1703.01365))
- Wang et al. 2017 — NIH ChestX-ray14 dataset
- Holistic AI P27 — XAI evaluation frameworks for clinical AI

---

## Citation

```bibtex
@software{awari2026xaimedical,
  author  = {Awari, Ajinkya},
  title   = {XAI Medical Imaging: Benchmarking Grad-CAM, SHAP, and Integrated Gradients on NIH ChestX-ray14},
  year    = {2026},
  url     = {https://github.com/ajinkya-awari/xai-medical-imaging-project-02},
  note    = {861 bounding box records, 8 pathologies, 90th-percentile IoU threshold}
}
```

---

> Not for clinical use. Benchmark results reflect model behaviour on NIH ChestX-ray14 under specific evaluation conditions — they do not represent diagnostic accuracy.

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d47a1,50:1a1f2e,100:0d1117&height=120&section=footer"/>
</div>
