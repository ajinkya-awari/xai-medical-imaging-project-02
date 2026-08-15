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

**Update rule:** One trace per bug, from discovery to verification; do not overwrite historical records.

### BUG-2026-08-15-2 - Native torch/torchvision mismatch in global interpreter (resolved by project venv)

- **Observed:** Contract tests pass, but loading the pre-existing checkpoint imports `torchvision` and fails with `RuntimeError: operator torchvision::nms does not exist` / a Windows native DLL load failure.
- **Evidence:** Environment reports `torch 2.13.0` and `torchvision 0.25.0+cpu`; `python -c "import torchvision"` reproduces the failure.
- **Impact:** The global interpreter cannot run model inference; the project-local runtime is unaffected. The NIH smoke remains pending because data is absent.
- **Resolution:** The global environment remains mismatched, but an isolated project `.venv` with `torch==2.12.1+cpu` and `torchvision==0.27.1+cpu` now imports successfully, loads the existing checkpoint, and passes a real synthetic FastAPI prediction.
- **Next action:** Use the isolated `.venv`; do not run the application from the mismatched global interpreter.
- **Scope:** No source repository, credentials, data, weights, or external artifact was changed to work around this issue.
