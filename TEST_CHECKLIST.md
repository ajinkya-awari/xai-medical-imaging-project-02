# TEST CHECKLIST - Project 01

This is the concrete done gate. Commands run from `E:\Projects\xai-medical-imaging` after the freeze is released unless marked planning-only.

## Planning-only scaffold checks

- [ ] `git status --short` shows the pre-existing `__results___files/` only plus intended control-plane files.
- [ ] `git check-ignore -q .claude/` succeeds.
- [ ] `python -m json.tool .claude/settings.json` succeeds.
- [ ] `bash -n .claude/hooks/pre-commit.sh .claude/hooks/lint-on-save.sh` succeeds where Bash is available.
- [ ] Secret scan finds no `WANDB_API_KEY`, `HF_TOKEN`, `ghp_`, `github_pat_`, or private clinical content.

## Day 5 - W&B gate

- [ ] `python -m pytest -q` passed before editing `src/train.py` (baseline had no tests and exited 1; focused tests were then added).
- [x] `rg -n "from src\\.config|import config|from config" src/train.py` recorded the verified `from src.config import CFG` import.
- [ ] A one-epoch/256-sample NIH smoke run logged the required metrics (not run: no local NIH fixture and no approved data path).
- [x] `WANDB_MODE=offline` was used for a secret-free one-epoch fixture; the later live check used a session-only approved key.
- [x] `wandb.finish()` executed on success and failure paths (offline run plus loader-failure fixture).
- [x] `.gitignore` already contained `.env`, `.env.*`, and `!.env.example` at baseline; `.env.example` was added with blank placeholders.
- [x] `wandb` was added to `requirements.txt` and installed as `wandb==0.28.2` for verification.
- [x] Focused contract tests pass: `4 passed`.

### Day 5 verification record - 2026-08-14

Observed offline fixture: `OFFLINE_TRAIN_RESULT=0.500000`, `OFFLINE_WANDB_RUN_FILES=1`; W&B summary contained `epoch`, `lr`, `phase`, `train_loss`, `train_auc`, `val_loss`, and `val_auc`.

Observed failure path: `FAILURE_PATH_EVENTS=['init', 'finish']`.

No `.env`, credentials, NIH data, model weights, or `__results___files/` changes were made.

### Current Day 5 blocker - 2026-08-15

- Local data check: `CSV present: False`; `PNG count: 0`.
- Online W&B check: completed; authenticated user `ajinkya18072001` and connectivity run `qwqgfql2` synced successfully.
- Safe stop: no data download, `.env` creation, credential commit, model training, or model artifact upload was performed.
- Resume condition: choose an approved smoke-data path, then run the bounded real NIH validation; do not treat the connectivity check as a training run.

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
- [ ] `docker compose build` and `docker compose up` succeed from a clean clone without local weights.

### Day 6 verification record - 2026-08-15

- `python -m pytest -q tests/test_inference_api_contract.py tests/test_day5_wandb_contract.py` -> `10 passed`.
- `python -m py_compile src/inference.py api/main.py app.py` -> passed; `git diff --check` -> passed.
- Docker verification -> not run: `docker` and `docker compose` are not installed in this environment.
- Global Python still has `torch 2.13.0` / `torchvision 0.25.0+cpu` native mismatch; the isolated project `.venv` resolves it with the matched CPU pair above.

## Day 7 - Streamlit/Space gate

- [x] Streamlit and FastAPI import the shared `src.inference` boundary.
- [ ] Public Space/model card links are verified; no restricted data is present.
- [x] README distinguishes local `/docs` from a hosted public API.

### Day 7 verification record - 2026-08-15

- README now documents local Streamlit/API commands, the local `/docs` boundary, checkpoint requirements, and the unclaimed external-deployment boundary.
- No public Space, model registry upload, hosted API, or model-card URL was created or claimed.

**Update rule:** Add the exact command and observed result for every new gate; do not mark a box from an agent summary alone.
