# BUG — Project 01

No active implementation bug is open. This file is a trace, not a generic backlog.

## Session verification - 2026-08-14

- No new bug opened during Day 5 W&B implementation.
- The first offline smoke command exposed only a Windows temporary-directory cleanup race while W&B's internal log handle was still open; the training and metric run completed. A persistent temporary directory plus explicit W&B teardown passed on rerun.

## Session verification - W&B onboarding pause - 2026-08-14

- No implementation bug opened.
- The online check was blocked by an invalid clipboard value, not by the Day 5 source implementation. The key was never written to the repository.
- The real-data smoke path is also blocked because `data\Data_Entry_2017.csv` and PNG images are absent locally.

## Session verification - W&B online auth resolved - 2026-08-15

- Windows clipboard content was unreliable when read via PowerShell's `Get-Clipboard` shortly after copying from the browser (observed lengths 0, 27, 75, and 122 characters with non-key content mixed in across several attempts), consistent with the OmniRoute clipboard interference noted earlier in the day. A direct paste into Notepad showed clean content, isolating the issue to the gap between copy and the `Get-Clipboard` read rather than the copy action itself.
- Fix: bypass `Get-Clipboard` entirely — paste directly into a masked `Read-Host -AsSecureString` prompt immediately after copying. This produced a clean, validating key on the next attempt.
- Separate incident: during troubleshooting, a candidate API key was pasted into chat by the user despite explicit instructions not to, and once landed in a PowerShell command line before the masked prompt appeared, risking persistence in the local PSReadLine history file. Remediation: the key was revoked on wandb.ai and the PowerShell history file (`(Get-PSReadLineOption).HistorySavePath`) was deleted before the successful, non-exposed attempt. No key value was ever committed, logged by the agent, or written to a repository file.

## Bug record template

```text
### BUG-YYYY-MM-DD-N — short title
- Found in / reproduction:
- Expected vs actual:
- Evidence (command/log/test):
- Hypotheses tested:
- Root cause:
- Fix and files:
- Regression test:
- Verification result:
- Commit / rollback:
```

Known planning hazards are recorded in `tasks/lessons.md` and `FINAL_VULNERABILITY_SCAN.md`; do not mark them fixed until runtime evidence exists.

### BUG-2026-08-15-3 — opencv-python (non-headless) survived through multiple commits (fixed)

- **Found in / reproduction:** `grep opencv requirements.txt` returns `opencv-python>=4.8.0` in HEAD. CLAUDE.md Rule 5 mandates `opencv-python-headless`. The Dockerfile added `libgl1` as a workaround but the root violation was never corrected.
- **Expected vs actual:** `requirements.txt` should contain `opencv-python-headless>=4.8.0`; actual was `opencv-python>=4.8.0` from the original repo baseline.
- **Evidence:** `git grep opencv` → one match in `requirements.txt`; Dockerfile contained `libgl1` apt install as the downstream symptom.
- **Root cause:** The non-headless package was present in the pre-existing repo before the Day 5/6 implementation; no commit audited it against Rule 5.
- **Fix and files:** `requirements.txt` line 15 changed to `opencv-python-headless>=4.8.0`; `Dockerfile` `libgl1` removed from the apt install list. API is identical for all project operations.
- **Regression test:** `python -m pytest -q` still returns `10 passed` after the change (headless does not affect Python-side API). Docker build verification pending Docker Desktop installation.
- **Verification result:** `git diff --check` PASS; test suite PASS. Docker build unverified until Docker Desktop available.
- **Commit / rollback:** Staged in this session's commit; no rollback risk (headless is a strict superset for server use).

**Update rule:** One trace per bug, from discovery to verification; do not overwrite historical records.

### BUG-2026-08-15-2 - Native torch/torchvision mismatch in global interpreter (resolved by project venv)

- **Observed:** Contract tests pass, but loading the pre-existing checkpoint imports `torchvision` and fails with `RuntimeError: operator torchvision::nms does not exist` / a Windows native DLL load failure.
- **Evidence:** Environment reports `torch 2.13.0` and `torchvision 0.25.0+cpu`; `python -c "import torchvision"` reproduces the failure.
- **Impact:** The global interpreter cannot run model inference; the project-local runtime is unaffected. The NIH smoke remains pending because data is absent.
- **Resolution:** The global environment remains mismatched, but an isolated project `.venv` with `torch==2.12.1+cpu` and `torchvision==0.27.1+cpu` now imports successfully, loads the existing checkpoint, and passes a real synthetic FastAPI prediction.
- **Next action:** Use the isolated `.venv`; do not run the application from the mismatched global interpreter.
- **Scope:** No source repository, credentials, data, weights, or external artifact was changed to work around this issue.
