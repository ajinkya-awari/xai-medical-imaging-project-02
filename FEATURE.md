# FEATURE - Project 01

The current implementation includes Day 5 W&B tracking and the local shared inference/FastAPI/Streamlit boundary. Docker packaging is present but not runtime-verified because Docker is not installed in the verification environment. External Hugging Face or hosted deployment remains unclaimed.

### FEAT-2026-08-14-1 - Day 5 W&B experiment tracking

- **Scope and user value:** Add reproducible W&B run configuration and scalar training/validation metrics without changing the training algorithm.
- **Approved spec/design section:** `DESIGN.md` section 4.3 and Day 5 plan; constrained by `FINAL_VULNERABILITY_SCAN.md` sections 1 and 3.
- **Files and interfaces:** `.env.example`, `requirements.txt`, `src/train.py`, and `tests/test_day5_wandb_contract.py`; `wandb.init`, two `wandb.log` calls, and `wandb.finish`.
- **Non-goals:** Grad-CAM logging, API/Docker work, HF uploads, live deployment, and model/data changes.
- **Plan and risk:** Protect secret files first; use exact `CFG` and metric names; validate offline because no approved credential is present.
- **Tests and expected output:** Four focused contract tests; offline one-epoch fixture with one W&B run file; failure-path finish check.
- **Implementation result:** Local implementation verified; online W&B authentication and a connectivity run were verified on 2026-08-15. The real one-epoch/256-sample NIH smoke remains pending because the approved NIH data path is absent.
- **Diff review and commit:** Complete staged diff reviewed line by line; committed as `feat: add W&B experiment tracking`.
- **Handoff:** Keep the session-only W&B key out of files; choose an approved NIH smoke-data path before claiming the real Day 5 gate.

### FEAT-2026-08-15-2 - Shared inference, FastAPI, and local serving boundary

- **Scope and user value:** Centralize preprocessing, checkpoint loading, sigmoid probabilities, Grad-CAM cleanup, and PNG encoding for the API and Streamlit surfaces.
- **Approved spec/design section:** `DESIGN.md` Day 6-7 sequence and the final vulnerability/impact reviews.
- **Files and interfaces:** `src/inference.py`, `api/main.py`, `app.py`, `Dockerfile`, `compose.yaml`, `requirements.txt`, and focused API contract tests.
- **Non-goals:** NIH data download, model training, random-weight fallback, public deployment, model upload, and external artifact publication.
- **Plan and risk:** Load a checkpoint once per API process when available; expose health/metadata without weights; reject malformed, non-PNG/JPEG, and over-10MB uploads.
- **Tests and expected output:** Focused inference/API plus Day 5 contract tests pass; Docker build/up remains unverified because Docker is unavailable locally.
- **Implementation result:** Implemented and committed as `bfee9ad`; Streamlit now uses the same inference boundary and no longer downloads weights automatically.
- **Diff review and commit:** Complete staged diff reviewed; no weights, data, credentials, or `__results___files/` were staged.
- **Handoff:** Repair the torch/torchvision environment before real checkpoint inference; then run the approved NIH smoke and live endpoint test.

## Feature record template

```text
### FEAT-YYYY-MM-DD-N - short title
- Scope and user value:
- Approved spec/design section:
- Files and interfaces:
- Non-goals:
- Plan and risk:
- Tests and expected output:
- Implementation result:
- Diff review and commit:
- Handoff:
```

**Update rule:** Keep one trace per feature and stop if implementation reality diverges from the approved design; update the spec and obtain approval before continuing.
