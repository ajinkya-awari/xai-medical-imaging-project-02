# TEST CHECKLIST - Project 01

This is the concrete done gate. Commands run from `E:\Projects\xai-medical-imaging` after the freeze is released unless marked planning-only.

## Planning-only scaffold checks

- [ ] `git status --short` shows the pre-existing `__results___files/` only plus intended control-plane files.
- [ ] `git check-ignore -q .claude/` succeeds.
- [ ] `python -m json.tool .claude/settings.json` succeeds.
- [ ] `bash -n .claude/hooks/pre-commit.sh .claude/hooks/lint-on-save.sh` succeeds where Bash is available.
- [ ] Secret scan finds no `WANDB_API_KEY`, `HF_TOKEN`, `ghp_`, `github_pat_`, or private clinical content.

## Day 5 - W&B gate

- [x] `python -m pytest -q` — `10 passed in 5.52s` (2026-08-15, .venv activated, all inference/API/W&B contract tests pass).
- [x] `rg -n "from src\\.config|import config|from config" src/train.py` recorded the verified `from src.config import CFG` import.
- [x] A one-epoch/256-sample NIH smoke run logged the required metrics — W&B run `zu1zp34y` (2026-08-16, Kaggle CUDA, 256 samples, warmup epoch only).
- [x] `WANDB_MODE=offline` was used for a secret-free one-epoch fixture; the later live check used a session-only approved key.
- [x] `wandb.finish()` executed on success and failure paths (offline run plus loader-failure fixture).
- [x] `.gitignore` already contained `.env`, `.env.*`, and `!.env.example` at baseline; `.env.example` was added with blank placeholders.
- [x] `wandb` was added to `requirements.txt` and installed as `wandb==0.28.2` for verification.
- [x] Focused contract tests pass: `4 passed`.

### Day 5 verification record - 2026-08-14

Observed offline fixture: `OFFLINE_TRAIN_RESULT=0.500000`, `OFFLINE_WANDB_RUN_FILES=1`; W&B summary contained `epoch`, `lr`, `phase`, `train_loss`, `train_auc`, `val_loss`, and `val_auc`.

Observed failure path: `FAILURE_PATH_EVENTS=['init', 'finish']`.

No `.env`, credentials, NIH data, model weights, or `__results___files/` changes were made.

### Day 5 NIH smoke verification - 2026-08-16 ✅ GATE CLOSED

- **Platform:** Kaggle notebook, CUDA (T4), NIH dataset pre-mounted, W&B key loaded via Kaggle Secrets only — never written to a file, command line, or chat.
- **Scope:** `smoke_train.py` — 256 samples total, 1 warmup epoch, `smoke_densenet121_chestxray.pth` output (production checkpoint not touched).
- **W&B run URL:** `https://wandb.ai/ajinkya18072001-university-college-london-ucl-/xai-medical-imaging/runs/zu1zp34y`
- **Metrics logged (from W&B dashboard):** `train_auc=0.55288`, `val_auc=0.55271`, `train_loss` and `val_loss` present, `epoch=1`, `phase=warmup`, `lr` recorded.
- **All required W&B keys confirmed present:** `epoch`, `lr`, `phase`, `train_loss`, `train_auc`, `val_loss`, `val_auc`.
- **Discrepancy to document honestly:** The script printed `Best val AUC=0.0000` at the end. This is a known scoping issue in `smoke_train.py`: the `best_auc` variable is updated only inside the finetune loop (`range(WARMUP_EPOCHS + 1, NUM_EPOCHS + 1)`), which is skipped when `NUM_EPOCHS=1` and `WARMUP_EPOCHS=1`. The warmup loop does not update `best_auc`, so the terminal print is misleading. W&B correctly received `val_auc=0.55271` via `wandb.log()` inside the warmup loop. The metric logging is correct; the terminal summary is not. This is a display-only issue in the smoke override and does not affect `src/train.py` production behaviour.
- **No NIH data, credentials, model weights, or `.env` were committed or uploaded to GitHub.**

### Day 5 online authentication verification - 2026-08-15

- Command: `python -c "import wandb; print(wandb.Api().viewer.username)"` — result: authenticated as `ajinkya18072001`. Key was loaded via `$env:WANDB_API_KEY` in the current PowerShell session only (masked `Read-Host -AsSecureString` prompt), never written to `.env`, a command line, chat, or any file.
- Command: `python -c "import wandb; run = wandb.init(project='xai-medical-imaging', job_type='connectivity-check'); wandb.log({'connectivity_check': 1}); print(run.url); run.finish()"` — result: live run synced successfully at `https://wandb.ai/ajinkya18072001-university-college-london-ucl-/xai-medical-imaging/runs/qwqgfql2`.
- An earlier candidate API key was accidentally exposed in chat during clipboard troubleshooting; it was revoked on wandb.ai and the local PowerShell history file was cleared before the successful attempt above.
- Remaining Day 5 blocker: only the real one-epoch/256-sample NIH smoke run — `CSV present: False`, `PNG count: 0`, no approved smoke-data path chosen. The Day 5 dashboard gate stays unchecked until that run is logged.

## Day 6 - API/Docker gate

- [x] Focused inference/API and Day 5 contract tests pass: `10 passed`.
- [x] TestClient checks `/health` and `/metadata` return 200 with the disclaimer and 14-label metadata.
- [x] Contract checks reject non-image and over-10MB uploads with 400; fake-model contract checks the all-label/top-prediction/confidence/base64 response shape.
- [x] API lifespan attempts one local checkpoint load per process and requires nested `model_state_dict`; no random-weight fallback exists.
- [x] Isolated `.venv` with `torch==2.12.1+cpu` and `torchvision==0.27.1+cpu` loaded the existing checkpoint; real FastAPI synthetic PNG check returned `200`, `model_loaded=True`, 14 labels, and a Grad-CAM payload.
- [x] `docker compose build` and `docker compose up -d` succeed; container Up on port 8000; `GET /health` returns `status=ok, model_loaded=True` (2026-08-16, local checkpoint mounted via volume).

### Day 6 verification record - 2026-08-15

- `python -m pytest -q` (2026-08-15 gate check): `10 passed in 5.52s` in isolated `.venv` with `torch==2.12.1+cpu` / `torchvision==0.27.1+cpu`.
- `git diff --check` -> PASS (no trailing whitespace; LF→CRLF autocrlf warnings are cosmetic).
- Docker verification -> not run: `docker --version` and `docker compose version` both return "not recognized" — Docker Desktop is not installed in this environment. Gate remains open until Docker Desktop is installed.
- `requirements.txt` corrected: `opencv-python` → `opencv-python-headless` (Rule 5 fix). Dockerfile `libgl1` apt package removed (headless no longer needs it). These are a single atomic change that must be verified by a Docker build once Docker Desktop is available.
- Global Python still has `torch 2.13.0` / `torchvision 0.25.0+cpu` native mismatch; the isolated project `.venv` resolves it with the matched CPU pair above.

### Day 6 Docker verification — 2026-08-16 ✅ GATE CLOSED

- **`docker compose build`:** Succeeded. Image built with `opencv-python-headless` and without `libgl1` (Rule 5 fix confirmed in container build).
- **`docker compose up -d`:** Succeeded. API container status: Up, port 8000 bound.
- **`GET http://localhost:8000/health`:** Returned `status=ok`, `model_loaded=True`. Model loaded from the mounted local checkpoint volume; no random-weight fallback triggered.
- **Discrepancy / honesty note:** The build runs with the local checkpoint mounted via `compose.yaml` volume. A fresh clone without the local weights file would start the container with `model_loaded=False` (health still returns 200, predict returns 503 — fail-closed per D-007). Docker Hub push not yet performed; `docker compose up --build` from a clone is the documented fallback per Rule 17.
- **No NIH data, credentials, model weights uploaded or committed.**

## Day 7 - Streamlit/Space gate

- [x] Streamlit and FastAPI import the shared `src.inference` boundary.
- [ ] Public Space/model card links are verified; no restricted data is present.
- [x] README distinguishes local `/docs` from a hosted public API.

### Day 7 verification record - 2026-08-15

- README now documents local Streamlit/API commands, the local `/docs` boundary, checkpoint requirements, and the unclaimed external-deployment boundary.
- No public Space, model registry upload, hosted API, or model-card URL was created or claimed.

**Update rule:** Add the exact command and observed result for every new gate; do not mark a box from an agent summary alone.
