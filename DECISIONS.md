# DECISIONS — XAI Medical Imaging / Project 01

This is a concise decision ledger, not a chat transcript. Add one entry for every meaningful technical or process choice.

## D-001 — Shared inference boundary is required before serving surfaces

- **Date / AI:** 2026-08-14 / Codex (GPT-5)
- **Decision:** Use `src/inference.py` as the single model-loading, preprocessing, prediction, and Grad-CAM boundary for Streamlit and FastAPI.
- **Alternatives:** duplicate loaders in `app.py` and `api/main.py`; expose raw model internals directly.
- **Why:** The source-specific scan found no existing `load_model()`/`run_inference()` helpers. One boundary prevents preprocessing, checkpoint-key, and label-order drift.
- **Verification:** Shared imports, `10 passed` contract tests, real checkpoint loading, and a real synthetic FastAPI prediction passed in the isolated `.venv`.

## D-002 — Local control plane is separate from public implementation

- **Date / AI:** 2026-08-14 / Codex (GPT-5)
- **Decision:** Keep `.claude/` gitignored and keep living records small, factual, and secret-free.
- **Why:** Agent instructions and session state are useful locally but are not product artifacts; public code must not contain credentials or restricted data.
- **Verification:** `.gitignore`, secret scan, and source diff audit before implementation.

## D-003 - Day 5 W&B tracking remains scalar and lifecycle-safe

- **Date / AI:** 2026-08-14 / Codex (GPT-5)
- **Decision:** Initialize one W&B run with the verified `CFG` fields, log the required scalar metrics in both warm-up and fine-tune loops, and always call `wandb.finish()` from a `finally` block. Do not add Grad-CAM logging to Day 5.
- **Alternatives:** log only the fine-tune loop; add a fixed validation image now; call `wandb.finish()` only on success.
- **Why:** The approved design requires both loops and scalar credibility, while the vulnerability review explicitly makes Grad-CAM optional and the checklist requires cleanup on failure.
- **Verification:** Focused AST contract tests pass; offline one-epoch fixture produced a local W&B run with `epoch`, `lr`, `phase`, `train_loss`, `train_auc`, `val_loss`, and `val_auc`; loader-failure fixture observed `init` then `finish`.

## D-004 - No real credential file during local verification

- **Date / AI:** 2026-08-14 / Codex (GPT-5)
- **Decision:** Commit only `.env.example`; do not create `.env` or run online W&B syncing without a user-approved credential.
- **Alternatives:** create a blank `.env`; use an ambient credential for a live run.
- **Why:** A blank or ambient credential would weaken the secret boundary and create an external side effect not required for secret-free local verification.
- **Verification:** `.env.example` contains only blank `WANDB_API_KEY=` and `HF_TOKEN=` placeholders; offline mode was used and no `.env` exists.

## D-005 - Pause after safe offline verification

- **Date / AI:** 2026-08-14 / Codex (GPT-5)
- **Decision:** Stop Day 5 after the documented offline verification and postpone live W&B validation until a clean API key and an approved smoke-data path are available. Do not download NIH data or save credentials in the repository.
- **Alternatives:** accept the malformed clipboard value; download the dataset immediately; substitute synthetic data as if it were the real acceptance smoke.
- **Why:** The local repository has no `data\Data_Entry_2017.csv` and no PNG images, and W&B rejected the loaded value as an invalid API key. Claiming the live gate would be misleading.
- **Verification:** `CSV present: False`, `PNG count: 0`; online `wandb.init()` failed with an invalid-key authentication error; no `.env`, data, model, or source-code changes were made.

## D-006 - Online W&B key loaded session-only, never persisted or shown

- **Date / AI:** 2026-08-15 / Claude (Sonnet 5)
- **Decision:** Load the W&B API key into `$env:WANDB_API_KEY` for the current PowerShell session only, via a masked `Read-Host -AsSecureString` prompt fed by the clipboard, rather than typing it into a command, `.env`, or any file. Verify auth via `wandb.Api().viewer.username` and a single throwaway connectivity run before considering any real smoke run.
- **Alternatives:** write the key to `.env`; pass it as a CLI argument; use the interactive `wandb login --relogin` prompt directly.
- **Why:** `.env`/CLI-argument approaches risk disk persistence (git, shell history); `wandb login --relogin`'s own prompt aborted in this terminal. The masked-prompt + session-env-var pattern keeps the key out of chat, files, and command history.
- **Verification:** `wandb.Api().viewer.username` returned `ajinkya18072001`; a connectivity run synced to `https://wandb.ai/ajinkya18072001-university-college-london-ucl-/xai-medical-imaging/runs/qwqgfql2`. No `.env` was created and `$env:WANDB_API_KEY` does not persist beyond the session.

## D-007 - Shared local inference boundary and fail-closed serving

- **Date / AI/model:** 2026-08-15 / Codex (GPT-5)
- **Decision:** Put checkpoint loading, preprocessing, sigmoid probabilities, Grad-CAM cleanup, and PNG encoding in `src/inference.py`; have FastAPI and Streamlit use it. Never fall back to random weights; keep health and metadata available when a checkpoint is missing or incompatible.
- **Alternatives:** duplicate inference code in each UI; auto-download weights; silently serve random predictions.
- **Why:** One boundary prevents label/preprocessing drift, while fail-closed prediction avoids presenting untrained outputs as medical results and keeps diagnostics available.
- **Verification:** `10 passed` focused/API and W&B contract tests; the isolated `.venv` loaded the existing checkpoint and returned a real synthetic FastAPI prediction with 14 labels and Grad-CAM; global interpreter mismatch remains documented.

## D-008 - Isolated matched CPU runtime for Project 01

- **Date / AI/model:** 2026-08-15 / Codex (GPT-5)
- **Decision:** Use the ignored project-local `.venv` with `torch==2.12.1+cpu` and `torchvision==0.27.1+cpu`; do not run Project 01 from the mismatched global interpreter.
- **Alternatives:** modify the global environment; bypass torchvision; use random weights for validation.
- **Why:** The matched pair loads the existing checkpoint and preserves the model architecture and Grad-CAM path without changing source behavior.
- **Verification:** Imports passed, checkpoint load passed, full test suite returned `10 passed`, and the real synthetic FastAPI request returned `200` with 14 labels and Grad-CAM.

## D-009 - Kaggle notebook is the approved NIH smoke path

- **Date / AI/model:** 2026-08-15 / Claude (coding-fallback)
- **Decision:** Use a Kaggle notebook to run the bounded 256-sample/1-epoch NIH smoke. The NIH dataset is pre-mounted at `/kaggle/input/datasets/organizations/nih-chest-xrays/data`, which `config.py` already detects automatically. The W&B API key is injected via Kaggle Secrets — never written to a file or command line. `smoke_train.py` overrides `MAX_SAMPLES=256`, `NUM_EPOCHS=1`, and `MODEL_FILENAME="smoke_densenet121_chestxray.pth"` without touching production code or the production checkpoint.
- **Alternatives:** download the full 40GB NIH dataset locally; use a synthetic fixture as a stand-in for real data; skip the smoke and claim the gate is closed.
- **Why:** Local NIH data is absent and will not be downloaded automatically (Rule: explicit user approval required). Kaggle has the dataset pre-mounted and free GPU. Synthetic fixtures were already used for contract tests but do not satisfy the real-data smoke criterion. The production checkpoint must not be overwritten by a 1-epoch smoke run.
- **Verification:** `smoke_train.py` exists in the repo; `config.py` Kaggle auto-detection confirmed. Gate remains open until the W&B run URL is recorded in `TEST_CHECKLIST.md`.

## D-010 - opencv-python-headless replaces opencv-python

- **Date / AI/model:** 2026-08-15 / Claude (coding-fallback)
- **Decision:** Replace `opencv-python>=4.8.0` with `opencv-python-headless>=4.8.0` in `requirements.txt`. Remove the `libgl1` apt package from the Dockerfile (headless does not need the GUI library). API is identical for all operations in the project (image read, resize, color conversion, overlay blending).
- **Alternatives:** keep `opencv-python` with the `libgl1` workaround (heavier image, non-standard); suppress the import error at runtime.
- **Why:** Non-negotiable Rule 5 from `CLAUDE.md`; headless is the correct choice for a slim server container. The `libgl1` apt package is only needed by the GUI renderer (OpenGL) which is never used here.
- **Verification:** `git diff --check` → PASS. Docker build with the updated pair cannot be run until Docker Desktop is installed; this change is safe to stage and commit now as the API is identical.

## Entry template

```text
### D-XXX — [decision]
- Date / AI/model:
- Decision:
- Alternatives:
- Why:
- Verification:
```

**Update rule:** Record the reason before or with the change, including the AI/model version that made the recommendation.
