---
title: T1 MLOps Stack
emoji: 🩻
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=T1%20MLOps%20Stack&fontSize=52&fontColor=fff&animation=twinkling&fontAlignY=38&desc=Production%20Layer%20for%20DenseNet121%20Chest%20X-ray%20Classifier&descAlignY=58&descAlign=50&descSize=16"/>

</div>

# T1 MLOps Stack

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi)
![Gradio](https://img.shields.io/badge/Gradio-4.0%2B-orange)
![Docker](https://img.shields.io/badge/Docker-compose-2496ED?logo=docker)
![W&B](https://img.shields.io/badge/W%26B-tracked-orange?logo=weightsandbiases)
![HF Model](https://img.shields.io/badge/HF%20Model-ajinkya1807%2Ft1--mlops--stack--model-yellow?logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green)

Experiment tracking, a Docker-hosted inference API, and Streamlit serving for
[ChestXplain](https://github.com/ajinkya-awari/xai-medical-imaging), a DenseNet121 chest X-ray
classifier trained on NIH ChestX-ray14.

ChestXplain trains to 0.769 mean AUC across 14 pathologies but had no way to track experiment
metrics, serve predictions through an API, or run reproducibly outside the training notebook.
This project adds those three layers and produces four verifiable public artifacts in the process.

---

## Artifacts

| Artifact | Status | Link / Evidence |
|---|---|---|
| GitHub | Live | [ajinkya-awari/xai-medical-imaging](https://github.com/ajinkya-awari/xai-medical-imaging) |
| W&B smoke run | Live | Run `zu1zp34y`, 256 samples, 1 epoch — train\_auc=0.553, val\_auc=0.553 |
| HF model repo | Live | [ajinkya1807/t1-mlops-stack-model](https://huggingface.co/ajinkya1807/t1-mlops-stack-model) |
| Docker CPU API | Verified locally | `docker compose up --build`, /health 200; no Docker Hub image published |
| HF Space demo | Live | [ajinkya1807/xai-medical-imaging](https://huggingface.co/spaces/ajinkya1807/xai-medical-imaging) |

---

## What was added

### W&B tracking

`src/train.py` logs loss, AUC, and learning rate each epoch in both the warmup and finetune
phases. Run `smoke_train.py` on Kaggle to verify the hooks work without pulling the full NIH
dataset locally. Smoke run `zu1zp34y` confirms it: `train_auc=0.55288`, `val_auc=0.55271`
on 256 samples. That number is not the model's real AUC; it just proves the tracking fires.

### Shared inference module

`src/inference.py` is the single place that handles checkpoint loading, image preprocessing,
sigmoid probabilities, Grad-CAM generation, and response encoding. Both `api/main.py` and
`app.py` import from it. There is no duplicated model logic between the two serving paths.

### FastAPI endpoint

`api/main.py` exposes `/health`, `/metadata`, and `POST /predict`. The model loads once at
FastAPI startup, not per request. Every prediction response includes all 14 label probabilities.
Uploads that are not PNG/JPEG or exceed 10 MB are rejected before the model is touched.

### Docker container

A `python:3.11-slim` image with CPU-only PyTorch installed as a dedicated layer before the rest
of the dependencies. No NIH data or model weights are baked in. The container expects the
checkpoint mounted at `models/` or pointed to via `MODEL_PATH`.

### HF model repository

The 28.5 MB DenseNet121 checkpoint is published at `ajinkya1807/t1-mlops-stack-model`. The
Streamlit app falls back to this when no local weights are found, pinned to commit `efa149c`
for reproducibility.

---

## About ChestXplain

ChestXplain is the classifier this project serves. The architecture is DenseNet121 pretrained
on ImageNet, fine-tuned on NIH ChestX-ray14 with a two-phase schedule: the backbone freezes
during warmup, then the full network trains in the finetune phase. The output layer is 14
independent sigmoid nodes, one per thoracic pathology. Grad-CAM targets the final dense block.

Full training baseline (20K images, 10 epochs, Kaggle T4 GPU): mean test AUC 0.769.
CheXNet on the full 112K dataset: 0.841.

Training references: Wang et al. 2017 (NIH ChestX-ray14), Huang et al. 2017 (DenseNet),
Selvaraju et al. 2017 (Grad-CAM), Rajpurkar et al. 2017 (CheXNet).

---

## Architecture

```text
NIH ChestX-ray14 (local only, never uploaded)
        |
        v
src/train.py ── scalar metrics ──► W&B run zu1zp34y  (smoke: train_auc=0.553, val_auc=0.553)
        |
        v
densenet121_chestxray.pth ──► HF model repo ajinkya1807/t1-mlops-stack-model
        |
        ├──► src/inference.py ◄── shared module (preprocessing + Grad-CAM)
        |          |
        |          ├──► api/main.py -> Docker CPU container (localhost:8000)
        |          |
        |          └──► app.py -> Gradio (HF Space + local)
        |
        └──► model card README.md (renders on HF)
```

---

## 1. W&B experiment tracking

`src/train.py` logs these scalars each epoch in both the warmup and finetune loops:

| Metric | Variable | Notes |
|---|---|---|
| `train/loss` | `tr_loss` | Mean BCE loss across training batches |
| `train/auc` | `tr_auc` | Mean AUC across 14 labels |
| `val/loss` | `va_loss` | Mean BCE loss across validation batches |
| `val/auc` | `va_auc` | Mean AUC across 14 labels |
| `learning_rate` | scheduler output | LR after ReduceLROnPlateau |
| `epoch` | loop counter | 1-indexed |

Smoke run `zu1zp34y`: 256 samples, 1 epoch. Verified: `train_auc=0.55288`, `val_auc=0.55271`.
These confirm the tracking infrastructure works, not that the model has converged. Full training
needs the complete 20K-image dataset.

---

## 2. Shared inference module

`src/inference.py` handles everything between raw input and model output:

| Function | What it does |
|---|---|
| `get_model_path()` | Resolves checkpoint from `MODEL_PATH` env var, local default, or HF Hub |
| `load_checkpoint_model()` | Loads `checkpoint["model_state_dict"]`; does not accept raw state dicts |
| `preprocess_image()` | PIL image to normalised batched tensor (ImageNet mean/std) |
| `predict_probabilities()` | Forward pass returning 14 sigmoid probabilities |
| `generate_gradcam_overlay()` | Grad-CAM on `DenseBlock4`; hooks cleaned up after each call |
| `probabilities_payload()` | 14-label dict with top prediction and confidence score |
| `run_inference()` | Probabilities + top-class Grad-CAM overlay + base64 PNG in one call |

Both `api/main.py` and `app.py` call these functions directly.

---

## 3. FastAPI inference endpoint

### Routes

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Model load status and research disclaimer |
| `GET` | `/metadata` | Label list, HF model repo, mean AUC, dataset, disclaimer |
| `POST` | `/predict` | PNG/JPEG up to 10 MB; returns all 14 probabilities + Grad-CAM |

### Response schema (`POST /predict`)

```json
{
  "predictions": [
    {"label": "Pneumothorax", "probability": 0.832},
    ...
  ],
  "top_prediction": "Pneumothorax",
  "confidence": 0.832,
  "gradcam_png_base64": "<base64 string>"
}
```

### Example

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@chest-xray.png"
```

API docs at `http://127.0.0.1:8000/docs`

Without a valid checkpoint, `/health` and `/metadata` return 200, but `/predict` returns 503.

> Not for clinical or diagnostic use. All responses include a `disclaimer` field.

---

## 4. Docker

### Build and start

```bash
docker compose up --build
```

No Docker Hub image is published. Build from source.

### What the image contains

- Base: `python:3.11-slim`
- CPU PyTorch installed in a separate layer before `requirements.txt`
- `opencv-python-headless` (no GUI libraries required in slim containers)
- `src/`, `api/`, `app.py` with no NIH data, no credentials, and no weights baked in

### Checkpoint

```bash
# compose.yaml handles this automatically:
volumes:
  - ./models:/app/models:ro
```

Or set `MODEL_PATH` to a path inside the container.

---

## 5. Gradio demo

### Run locally

```bash
python app.py
```

### Live Space

[huggingface.co/spaces/ajinkya1807/xai-medical-imaging](https://huggingface.co/spaces/ajinkya1807/xai-medical-imaging)

The app loads the local checkpoint at `models/densenet121_chestxray.pth` if present. If not,
it downloads from `ajinkya1807/t1-mlops-stack-model` (pinned to commit `efa149c`).

Upload limit: 20 million pixels decoded. The Grad-CAM overlay is generated for the top
predicted pathology. Do not upload patient-identifiable or restricted clinical images.


---

## Verification gates

| Gate | Status | Evidence |
|---|---|---|
| W&B tracking | Closed | Run `zu1zp34y`, train_auc=0.553, val_auc=0.553 |
| Docker build + /health | Closed | `compose build + up`, /health 200 |
| HF model repository | Closed | [commit efa149c](https://huggingface.co/ajinkya1807/t1-mlops-stack-model/commit/efa149c) |
| HF Space demo | Closed | [ajinkya1807/xai-medical-imaging](https://huggingface.co/spaces/ajinkya1807/xai-medical-imaging) |

---

## Setup

### Prerequisites

Python 3.9 or newer. 8 GB RAM minimum (16 GB recommended for full training). Docker Desktop
for the container API path.

### Clone

```bash
git clone https://github.com/ajinkya-awari/xai-medical-imaging.git
cd xai-medical-imaging
```

### Install

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install --upgrade pip
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

For GPU, replace the PyTorch install with the appropriate CUDA pair from
[pytorch.org](https://pytorch.org/get-started/locally/).

### Run the Streamlit app

```bash
streamlit run app.py
```

### Run the local API

```bash
uvicorn api.main:app --reload
```

Needs a checkpoint at `models/densenet121_chestxray.pth` or via `MODEL_PATH`. Without one,
`/predict` returns 503.

### Build the Docker container

```bash
docker compose up --build
```

---

## Reproducing the smoke run on Kaggle

The smoke run uses 256 NIH images and 1 epoch. No local data download needed.

1. Open [kaggle.com](https://kaggle.com) and create a new notebook.
2. Add dataset: **+ Add Data**, search `NIH Chest X-rays` (by `nih-chest-xrays`), add it.
3. Add your W&B key: Notebook sidebar, **Secrets**, add `WANDB_API_KEY`, toggle
   **Attach to notebook** on.
4. In the first cell:

```python
from kaggle_secrets import UserSecretsClient
import os
os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")

!git clone https://github.com/ajinkya-awari/t1-mlops-stack.git
%cd t1-mlops-stack
!pip install -q -r requirements.txt
```

5. Run the smoke:

```python
!python smoke_train.py
```

Do not increase `MAX_SAMPLES` or `NUM_EPOCHS` without a specific reason. The purpose of the
smoke is to verify the tracking hooks, not to measure AUC.

---

## Repository layout

```
t1-mlops-stack/
├── src/
│   ├── config.py        # Hyperparameters, paths, disease labels
│   ├── dataset.py       # Data loading, transforms, train/val/test split
│   ├── model.py         # DenseNet121 with freeze/unfreeze utilities
│   ├── inference.py     # Preprocessing, prediction, Grad-CAM, encoding
│   ├── train.py         # Two-phase training loop with W&B logging
│   ├── evaluate.py      # Test AUC computation and ROC curves
│   ├── gradcam.py       # Grad-CAM implementation
│   ├── visualize.py     # Grad-CAM sample grid
│   └── __init__.py
├── api/
│   └── main.py          # FastAPI: /health, /metadata, POST /predict
├── tests/
│   ├── test_day5_wandb_contract.py
│   └── test_inference_api_contract.py
├── outputs/             # Evaluation figures and test results
├── app.py               # Streamlit app
├── smoke_train.py       # 256-sample smoke for W&B gate
├── run_all.py           # Train, evaluate, visualise
├── Dockerfile
├── compose.yaml
├── requirements.txt
├── packages.txt         # System packages for Streamlit build
└── README.md
```

---

## Citation

```bibtex
@software{awari2026t1mlops,
  author  = {Awari, Ajinkya},
  title   = {T1 MLOps Stack: Serving and Verification for ChestXplain},
  year    = {2026},
  url     = {https://github.com/ajinkya-awari/xai-medical-imaging},
  license = {MIT}
}
```

---

> Research prototype. Not for clinical or diagnostic use. Do not use predictions as a substitute
> for professional medical evaluation.

---

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer"/>
