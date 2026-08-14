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
- [ ] A one-epoch/256-sample NIH smoke run logged the required metrics (not run: no local NIH fixture and no data download authorized).
- [x] `WANDB_MODE=offline` was used for a secret-free one-epoch fixture; no live run was attempted without an approved credential.
- [x] `wandb.finish()` executed on success and failure paths (offline run plus loader-failure fixture).
- [x] `.gitignore` already contained `.env`, `.env.*`, and `!.env.example` at baseline; `.env.example` was added with blank placeholders.
- [x] `wandb` was added to `requirements.txt` and installed as `wandb==0.28.2` for verification.
- [x] Focused contract tests pass: `4 passed`.

### Day 5 verification record - 2026-08-14

Observed offline fixture: `OFFLINE_TRAIN_RESULT=0.500000`, `OFFLINE_WANDB_RUN_FILES=1`; W&B summary contained `epoch`, `lr`, `phase`, `train_loss`, `train_auc`, `val_loss`, and `val_auc`.

Observed failure path: `FAILURE_PATH_EVENTS=['init', 'finish']`.

No `.env`, credentials, NIH data, model weights, or `__results___files/` changes were made.

### Current Day 5 blocker - 2026-08-14

- Local data check: `CSV present: False`; `PNG count: 0`.
- Online W&B check: not completed; `wandb.init()` rejected the loaded clipboard value as an invalid API key.
- Safe stop: no data download, `.env` creation, credential commit, model training, or Day 6 work was performed.
- Resume condition: load a clean W&B API key without exposing it, choose an approved smoke-data path, then run the bounded online validation.

## Day 6 - API/Docker gate (future)

- [ ] `python -m pytest -q` and focused API tests pass.
- [ ] `/health` and `/metadata` return 200 with the disclaimer and correct 14-label metadata.
- [ ] `/predict` accepts PNG/JPEG, rejects non-image/over-10MB/unreadable uploads with 400, and returns all 14 probabilities plus top prediction, confidence, and base64 Grad-CAM.
- [ ] Model loads once in lifespan; checkpoint uses `model_state_dict`.
- [ ] `docker compose build` and `docker compose up` succeed from a clean clone without local weights.

## Day 7 - Streamlit/Space gate (future)

- [ ] Streamlit and FastAPI use the same inference boundary.
- [ ] Public Space/model card links are verified; no restricted data is present.
- [ ] README distinguishes local `/docs` from a hosted public API.

**Update rule:** Add the exact command and observed result for every new gate; do not mark a box from an agent summary alone.
