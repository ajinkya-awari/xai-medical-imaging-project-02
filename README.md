<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=XAI%20Medical%20Imaging&fontSize=52&fontColor=fff&animation=twinkling&fontAlignY=38&desc=Grad-CAM%20%2B%20SHAP%20%2B%20Integrated%20Gradients%20on%20NIH%20Chest%20X-rays&descAlignY=58&descAlign=50&descSize=16"/>

</div>

# XAI Medical Imaging

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-orange)
![SHAP](https://img.shields.io/badge/SHAP-0.44%2B-blueviolet)
![Captum](https://img.shields.io/badge/Captum-0.7%2B-green)
![Dataset](https://img.shields.io/badge/Dataset-NIH%20ChestX--ray14-yellow)
![HF Space](https://img.shields.io/badge/HF%20Space-Live-brightgreen?logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green)

Quantitative XAI benchmark for chest X-ray classification. Three explanation methods — Grad-CAM, SHAP GradientExplainer, and Integrated Gradients — evaluated against NIH ChestX-ray14 ground-truth bounding box annotations using Intersection-over-Union (IoU).

This is the evaluation layer on top of [T1 MLOps Stack](https://github.com/ajinkya-awari/xai-medical-imaging): the same DenseNet121 checkpoint, with three explanation heads added and benchmarked against clinical annotations. The project targets Holistic AI's P27 XAI evaluation framework.

---

## Artifacts

| Artifact | Status | Link |
|---|---|---|
| GitHub | Live | [ajinkya-awari/xai-medical-imaging-project-02](https://github.com/ajinkya-awari/xai-medical-imaging-project-02) |
| HF model repo | Live | [ajinkya1807/t1-mlops-stack-model](https://huggingface.co/ajinkya1807/t1-mlops-stack-model) |
| HF Space demo | Live | [ajinkya1807/xai-medical-imaging](https://huggingface.co/spaces/ajinkya1807/xai-medical-imaging) |
| IoU benchmark | **Complete (3/3)** | Grad-CAM: **0.1289** · IG: **0.0792** · SHAP: **0.0704** mean IoU vs NIH BBox |

---

## XAI Methods

Three explanation methods share a unified interface: `.explain(image_tensor, class_idx) → (heatmap_2D, overlay_rgb)`.

| Method | Implementation | Speed | Notes |
|---|---|---|---|
| Grad-CAM | `src/gradcam.py` wrapped by `GradCAMExplainer` | ~10 ms/image | Gradient backprop through DenseBlock4 |
| SHAP | `SHAPExplainer` via `shap.GradientExplainer` | ~100–200 ms/image | 10-image background from training split |
| Integrated Gradients | `IGExplainer` via Captum | ~50 ms/image | 50 steps, zero baseline |

All three are benchmarked with the same 90th-percentile threshold IoU metric against the 8 NIH-annotated pathologies.

---

## IoU Benchmark Results

IoU computed against NIH ChestX-ray14 bounding box annotations (8 pathologies, 861 annotated records). Attribution maps thresholded at the 90th percentile; scaled from 1024×1024 image space to 224×224 model input.

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

861 annotated records · 8 pathologies · 90th-percentile threshold · NIH ChestX-ray14 BBox annotations · scaled from 1024×1024 to 224×224.
Cardiomegaly scores highest across all three methods (large, well-defined bounding box). Nodule is hardest (small, diffuse lesion). Grad-CAM leads overall, consistent with the literature on gradient saturation in multi-label classifiers.

---

## Architecture

```text
NIH ChestX-ray14 (local only, never uploaded)
        |
        v
densenet121_chestxray.pth ──► HF model repo ajinkya1807/t1-mlops-stack-model
        |
        v
src/inference.py (load_checkpoint_model, preprocess_image)
        |
        ├──► src/explainers.py
        │        ├── GradCAMExplainer   (wraps src/gradcam.py)
        │        ├── SHAPExplainer      (shap.GradientExplainer)
        │        └── IGExplainer        (captum IntegratedGradients)
        |
        ├──► src/xai_bench.py    ── IoU vs NIH BBox annotations → outputs/xai/*.json
        ├──► src/xai_compare.py  ── 10-image × 3-method comparison grid → outputs/xai/*.png
        └──► app.py              ── Gradio Space with method selector
```

---

## Features

### Unified explainer interface

`src/explainers.py` defines three classes behind a single contract:
`.explain(image_tensor, class_idx)` returns `(heatmap_2D, overlay_rgb)` where `heatmap_2D` is a normalised `(H, W)` float array and `overlay_rgb` is a `(H, W, 3)` uint8 overlay.

### IoU benchmark

`src/xai_bench.py` evaluates each method against 861 NIH bounding box records across 8 pathologies. Bounding box coordinates are scaled from the 1024×1024 image space to 224×224. The 90th-percentile threshold selects the brightest 10% of each attribution map for comparison with the ground-truth box.

### Comparison figures

`src/xai_compare.py` generates a 10-image × 4-column grid (original + 3 methods) using `random.seed(42)` for reproducibility.

### Interactive demo

`app.py` is a Gradio Space with a method selector (Grad-CAM / SHAP / Integrated Gradients). Only the selected explainer is instantiated at runtime to stay within HF Space memory limits.

---

## Setup

### Prerequisites

Python 3.9 or newer. CUDA recommended for SHAP and IG benchmarks.

### Clone

```bash
git clone https://github.com/ajinkya-awari/xai-medical-imaging-project-02.git
cd xai-medical-imaging-project-02
```

### Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1

pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Run the demo locally

```bash
python app.py
```

Loads `models/densenet121_chestxray.pth` if present; otherwise downloads from HF Hub.

### Run the IoU benchmark (requires NIH data)

```bash
python -m src.xai_bench
```

Set `CFG.DATA_DIR`, `CFG.BBOX_PATH`, and `CFG.MODEL_DIR` in `src/config.py` before running.

---

## Repository layout

```
xai-medical-imaging-project-02/
├── src/
│   ├── config.py        # Paths, disease labels, XAI settings
│   ├── explainers.py    # GradCAMExplainer, SHAPExplainer, IGExplainer
│   ├── xai_bench.py     # IoU benchmark against NIH BBox annotations
│   ├── xai_compare.py   # 10-image × 3-method comparison figures
│   ├── inference.py     # Checkpoint loading, preprocessing
│   ├── gradcam.py       # Grad-CAM implementation (unchanged from T1)
│   ├── dataset.py       # Data loading for background tensor sampling
│   ├── model.py         # DenseNet121 architecture
│   └── train.py, evaluate.py, visualize.py
├── api/
│   └── main.py          # FastAPI /health, /metadata, POST /predict
├── tests/
│   ├── test_explainers_synthetic.py
│   ├── test_inference_api_contract.py
│   └── test_app_wiring.py
├── notebooks/
│   └── kaggle_xai_triple.ipynb   # Full benchmark notebook for Kaggle T4 GPU
├── outputs/xai/         # IoU JSON results + comparison figures (generated)
├── app.py               # Gradio Space with method selector
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Related work

- Grad-CAM: Selvaraju et al. 2017 ([arXiv:1610.02055](https://arxiv.org/abs/1610.02055))
- SHAP: Lundberg & Lee 2017 ([arXiv:1705.07874](https://arxiv.org/abs/1705.07874))
- Integrated Gradients: Sundararajan et al. 2017 ([arXiv:1703.01365](https://arxiv.org/abs/1703.01365))
- NIH ChestX-ray14: Wang et al. 2017

---

## Citation

```bibtex
@software{awari2026xaimedical,
  author  = {Awari, Ajinkya},
  title   = {XAI Medical Imaging: Benchmarking Grad-CAM, SHAP, and Integrated Gradients on NIH ChestX-ray14},
  year    = {2026},
  url     = {https://github.com/ajinkya-awari/xai-medical-imaging-project-02},
  license = {MIT}
}
```

---

> Research prototype. Not for clinical or diagnostic use. Do not use predictions as a substitute for professional medical evaluation.

---

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer"/>
