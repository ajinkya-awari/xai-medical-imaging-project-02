---
title: T1 MLOps Stack — ChestXplain
emoji: 🩻
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: app.py
pinned: false
---

# T1 MLOps Stack

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![Docker](https://img.shields.io/badge/Docker-compose-2496ED?logo=docker)
![W&B](https://img.shields.io/badge/W%26B-tracked-orange?logo=weightsandbiases)
![HF Model](https://img.shields.io/badge/HF%20Model-ajinkya1807%2Ft1--mlops--stack--model-yellow?logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green)

**MLOps serving and verification stack built on top of ChestXplain — an explainable chest X-ray classifier.**

A trained model with no production signal is a research artifact, not an engineering asset.
This project adds three production layers to ChestXplain in three days:
W&B experiment tracking, a Docker-hosted FastAPI inference API, and a Hugging Face Space demo
(source-prepared; Space not yet created) — producing verifiable public artifacts that demonstrate
end-to-end ML operationalisation.

---

## Public Artifacts

| Artifact | Status | Link / Evidence |
|---|---|---|
| GitHub repository | ✅ Live | [ajinkya-awari/t1-mlops-stack](https://github.com/ajinkya-awari/t1-mlops-stack) |
| W&B smoke run | ✅ Live | Run `zu1zp34y` — train_auc=0.553, val_auc=0.553 (256 samples, 1 epoch) |
| HF model repository | ✅ Live | [ajinkya1807/t1-mlops-stack-model](https://huggingface.co/ajinkya1807/t1-mlops-stack-model) |
| Docker CPU API | ✅ Verified locally | `docker compose up --build` — `/health` 200; no Docker Hub image published |
| HF Space (Streamlit) | ❌ Not created | Source prepared; Space has not been created yet |

---

## What This Project Adds

ChestXplain already trains to a mean test AUC of 0.769 across 14 chest pathologies
(full 20K-image training run, Kaggle T4 GPU). What it lacked was any production signal:
no tracked experiment metrics, no reproducible inference API, and no public interactive demo.

**T1 MLOps Stack** introduces:

- **W&B experiment tracking** — scalar metrics (loss, AUC, learning rate) logged per epoch in both
  the warm-up and fine-tune phases of the two-phase training loop; reproducible from Kaggle with
  a single smoke command. Smoke gate verified: run `zu1zp34y`, train_auc=0.553, val_auc=0.553
  (256 samples, 1 epoch, tracking infrastructure confirmed).
- **Shared inference boundary** (`src/inference.py`) — a single module that owns checkpoint
  resolution, `model_state_dict` loading, preprocessing, probability computation, Grad-CAM
  generation, and PNG encoding. Both FastAPI and Streamlit call the same functions; there is no
  duplicated model logic.
- **FastAPI inference API** — `/health`, `/metadata`, and `POST /predict` endpoints; the model
  loads once at startup via FastAPI's `lifespan`; all 14 label probabilities are returned per
  request; non-image and oversized uploads are rejected with typed errors.
- **Docker deployment** — a reproducible CPU API container built from `python:3.11-slim`; the
  image installs CPU PyTorch before the remaining dependencies and does not bundle NIH data or
  model weights; verified locally via `docker compose up --build`.
- **Hugging Face model repository** — the approved DenseNet121 checkpoint (28.5 MB, SHA-256
  verified) is published to `ajinkya1807/t1-mlops-stack-model` under MIT license; the model
  card documents training conditions, AUC results, and the research-only disclaimer.
- **HF Space (Streamlit)** — source-prepared for CPU deployment; the app falls back to the public
  HF model checkpoint when no local weights are available, enforces a 10 MB / 20 Mpx upload cap,
  and limits Grad-CAM rendering to four classes to keep CPU inference responsive.

---

## About ChestXplain (the application being operationalised)

ChestXplain is the underlying XAI application this project wraps with production infrastructure.
It is a DenseNet121 classifier with Grad-CAM explanations trained on the NIH ChestX-ray14 dataset.

**Baseline results (ChestXplain full training — 20K images, 10 epochs, Kaggle T4 GPU):**
Mean test AUC: **0.769** across 14 thoracic pathology labels.
CheXNet benchmark (full 112K dataset): 0.841.

The model checkpoint published to the HF model repository (`densenet121_chestxray.pth`, 28.5 MB)
is the approved ChestXplain baseline artifact. Architecture: DenseNet121 pretrained on ImageNet,
AdaptiveAvgPool → Dropout(0.3) → Linear(1024, 14), sigmoid output.
Explainability: Grad-CAM on the final dense block (`DenseBlock4`).

> ChestXplain references: Wang et al. (2017) NIH ChestX-ray14; Huang et al. (2017) DenseNet121;
> Selvaraju et al. (2017) Grad-CAM; Rajpurkar et al. (2017) CheXNet.

---

## System Architecture

```text
NIH ChestX-ray14 (local only — never uploaded)
        │
        ▼
src/train.py ── scalar metrics ──► W&B run zu1zp34y (smoke: train_auc=0.553, val_auc=0.553)
        │
        ▼
densenet121_chestxray.pth ──► HF model repo ajinkya1807/t1-mlops-stack-model
        │
        ├──► src/inference.py ◄── shared boundary (one preprocessing + Grad-CAM path)
        │          │
        │          ├──► api/main.py → Docker CPU container (localhost:8000)
        │          │
        │          └──► app.py → Streamlit → HF Space (not yet created)
        │
        └──► model card README.md (renders on HF)
```

---

## 1 · W&B Experiment Tracking

`src/train.py` initialises one W&B run per training session and logs the following scalars each
epoch inside both the warm-up and fine-tune loops:

| Metric | Variable | Description |
|---|---|---|
| `train/loss` | `tr_loss` | Mean BCE loss over training batches |
| `train/auc` | `tr_auc` | Mean AUC across 14 labels (train) |
| `val/loss` | `va_loss` | Mean BCE loss over validation batches |
| `val/auc` | `va_auc` | Mean AUC across 14 labels (validation) |
| `learning_rate` | scheduler output | Current LR after ReduceLROnPlateau |
| `epoch` | loop counter | 1-indexed epoch number |

**Smoke gate (Day 5):** W&B run `zu1zp34y`, 256 samples, 1 epoch.
Verified metrics: `train_auc=0.55288`, `val_auc=0.55271`.
These confirm the tracking infrastructure works; they are not indicative of full-training AUC
(which requires the full 20K-image ChestXplain training run).

---

## 2 · Shared Inference Boundary

`src/inference.py` is the single source of truth for all model-facing operations:

| Function | Purpose |
|---|---|
| `get_model_path()` | Resolve checkpoint from `MODEL_PATH` env var, local default, or HF Hub |
| `load_checkpoint_model()` | Load `checkpoint["model_state_dict"]`; never accept raw state dicts or random weights |
| `preprocess_image()` | PIL → normalised (ImageNet stats) → batched tensor |
| `predict_probabilities()` | Model forward pass → 14 independent sigmoid probabilities |
| `generate_gradcam_overlay()` | Grad-CAM on `DenseBlock4`; hooks cleaned up after each call |
| `probabilities_payload()` | Stable 14-label JSON payload for both API and Streamlit |
| `run_inference()` | Convenience wrapper: probabilities + top-class Grad-CAM + base64 PNG |

Both `api/main.py` and `app.py` call this module. Model logic is not duplicated.

---

## 3 · FastAPI Inference API

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Returns model status, load timestamp, and research disclaimer |
| `GET` | `/metadata` | Returns label list, HF model repo URL, mean AUC, dataset, and disclaimer |
| `POST` | `/predict` | Accepts PNG/JPEG ≤ 10 MB; returns all 14 label probabilities + Grad-CAM |

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

### Example request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@path/to/chest-xray.png"
```

Interactive API docs: `http://127.0.0.1:8000/docs`

The model loads once at FastAPI startup via `lifespan`. Without a valid checkpoint, `/health` and
`/metadata` return 200; `/predict` returns `503 Model checkpoint is unavailable`.

> **Research disclaimer**: this API is not approved for clinical or diagnostic use.
> All responses include a `disclaimer` field. Do not use predictions as medical advice.

---

## 4 · Docker Deployment

### Build and run

```bash
docker compose up --build
```

> No Docker Hub image has been published. Build locally from source.

### What the image contains

- `python:3.11-slim` base
- CPU-only PyTorch (`torch==2.12.1`, `torchvision==0.27.1`) installed in a dedicated layer
- `opencv-python-headless` (no GUI libraries required in slim containers)
- `src/`, `api/`, `app.py` — no NIH data, credentials, or model weights

### Supplying the checkpoint

Mount the approved local checkpoint before starting:

```bash
# compose.yaml does this automatically:
volumes:
  - ./models:/app/models:ro
```

The `MODEL_PATH` environment variable can also point to an explicit path inside the container.

---

## 5 · HuggingFace Space (Streamlit)

### Local use

```bash
streamlit run app.py
```

Open `http://localhost:8501`. The app loads the local checkpoint at
`models/densenet121_chestxray.pth` if present, or downloads the approved public artifact from
`ajinkya1807/t1-mlops-stack-model` (pinned commit `efa149c`) if no local weights are found.

### Space constraints

- **Upload limit**: 10 MB encoded; 20 million pixels decoded.
- **Grad-CAM cap**: top 4 predictions only (prevents 14 sequential backward passes on CPU).
- **Disclaimer**: a research-only warning is shown before and after every inference.

> The HF Space has not been created yet. The source is prepared and the app runs locally.

---

## Verification Gates

| Gate | Status | Evidence |
|---|---|---|
| W&B experiment tracking | ✅ CLOSED | Run `zu1zp34y`, train_auc=0.553, val_auc=0.553, Day 5 |
| Docker build + `/health` 200 | ✅ CLOSED | `compose build + up`, Day 6 |
| HF model repository | ✅ CLOSED | [commit efa149c](https://huggingface.co/ajinkya1807/t1-mlops-stack-model/commit/efa149c), Day 7 |
| HF Space (Streamlit) | ❌ PENDING | Source prepared; Space not yet created |

---

## Setup & Usage

### Prerequisites

- Python 3.9 or newer
- 8 GB RAM minimum (16 GB recommended for full training)
- GPU optional but significantly speeds up training
- Docker Desktop (for the container API path)

### Step 1 — Clone

```bash
git clone https://github.com/ajinkya-awari/t1-mlops-stack.git
cd t1-mlops-stack
```

### Step 2 — Install CPU PyTorch first

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cpu
```

For CUDA, substitute your matched pair from the [PyTorch installer](https://pytorch.org/get-started/locally/).

### Step 3 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Launch the Streamlit app

```bash
streamlit run app.py
```

The app downloads the approved checkpoint from the HF model repository on first launch if no
local weights are present.

### Step 5 — Launch the local API

```bash
uvicorn api.main:app --reload
```

The API requires a local checkpoint at `models/densenet121_chestxray.pth` or a path set via
`MODEL_PATH`. Without one, `/predict` returns 503.

### Step 6 — Build the Docker API container

```bash
docker compose up --build
```

---

## Reproducing the W&B Smoke Run (Kaggle)

The approved smoke run uses 256 NIH images and 1 epoch — no local data download needed.

1. Go to [kaggle.com](https://kaggle.com) → New Notebook.
2. Add dataset: **+ Add Data** → search `NIH Chest X-rays` (by `nih-chest-xrays`) → Add.
3. Add W&B key: Notebook sidebar → **Secrets** → Add `WANDB_API_KEY` → toggle **Attach to notebook** ON.
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

Copy the W&B run URL from the output and retain it with the experiment record.
Do not increase `MAX_SAMPLES` or `NUM_EPOCHS` without explicit approval.

---

## Repository Structure

```
t1-mlops-stack/
├── src/
│   ├── config.py          # Hyperparameters, paths, disease labels
│   ├── dataset.py         # Data loading, transforms, train/val/test split
│   ├── model.py           # DenseNet121 architecture + freeze/unfreeze utilities
│   ├── inference.py       # Shared: preprocessing, prediction, Grad-CAM, encoding
│   ├── train.py           # Two-phase training loop with W&B logging + checkpointing
│   ├── evaluate.py        # Test set evaluation, AUC computation, ROC curves
│   ├── gradcam.py         # Grad-CAM implementation + overlay generation
│   ├── visualize.py       # Sample Grad-CAM grid generation
│   └── __init__.py
├── api/
│   └── main.py            # FastAPI: /health, /metadata, POST /predict
├── tests/
│   ├── test_day5_wandb_contract.py   # W&B config contract
│   └── test_inference_api_contract.py # API schema contract
├── outputs/
│   ├── auc_barplot.png    # Per-class AUC bar chart (ChestXplain baseline)
│   ├── roc_curves.png     # ROC curves for all 14 diseases
│   ├── gradcam_samples.png # Grad-CAM on real NIH X-rays
│   └── test_results.json  # Full AUC numbers
├── app.py                 # Streamlit web application (HF Space entrypoint)
├── smoke_train.py         # 256-sample / 1-epoch W&B gate script
├── run_all.py             # Master script: train → evaluate → visualise
├── Dockerfile             # CPU API image (python:3.11-slim)
├── compose.yaml           # Local API composition
├── requirements.txt       # Dependencies (CPU PyTorch wheels pre-declared)
├── packages.txt           # System packages for HF Space build
└── README.md
```

---

## Citation

```bibtex
@software{awari2026t1mlops,
  author    = {Awari, Ajinkya},
  title     = {T1 MLOps Stack: Serving and Verification for ChestXplain},
  year      = {2026},
  url       = {https://github.com/ajinkya-awari/t1-mlops-stack},
  license   = {MIT}
}
```

---

## Disclaimer

This system is a **research prototype** and is **not** intended for clinical diagnostic use.
Predictions should not replace professional medical evaluation.
Always consult a qualified radiologist for diagnosis.
