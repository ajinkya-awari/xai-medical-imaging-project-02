# BUG — Project 01

No active implementation bug is open. This file is a trace, not a generic backlog.

## Session verification - 2026-08-14

- No new bug opened during Day 5 W&B implementation.
- The first offline smoke command exposed only a Windows temporary-directory cleanup race while W&B's internal log handle was still open; the training and metric run completed. A persistent temporary directory plus explicit W&B teardown passed on rerun.

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
