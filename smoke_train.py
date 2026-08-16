"""
smoke_train.py — bounded 256-sample / 1-epoch W&B smoke for Project 01.

Purpose
-------
This script runs the minimum approved smoke to close the Day 5 NIH gate:
  - 256 samples total (217 train+val / 39 test after CFG split)
  - 1 epoch (warmup loop only)
  - Saves to smoke_densenet121_chestxray.pth — never overwrites production weights
  - Logs epoch, lr, phase, train_loss, train_auc, val_loss, val_auc to W&B

Run on Kaggle (recommended — NIH dataset is pre-mounted, no local download needed)
---------------------------------------------------------------------------
Step 1: Create a new Kaggle notebook.

Step 2: Add the NIH dataset:
        Notebook sidebar → + Add Data → search "NIH Chest X-rays" (by nih-chest-xrays) → Add
        This mounts the dataset at /kaggle/input/datasets/organizations/nih-chest-xrays/data
        which config.py detects automatically.

Step 3: Add your W&B API key as a Kaggle Secret:
        Notebook sidebar → Secrets → Add New Secret
        Name: WANDB_API_KEY
        Value: <your key from wandb.ai/settings>
        Toggle "Attach to notebook" ON.

Step 4: In the first notebook cell, load the secret and clone the repo:

    from kaggle_secrets import UserSecretsClient
    import os
    os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")

    # Clone and install
    !git clone https://github.com/ajinkya-awari/t1-mlops-stack.git
    %cd t1-mlops-stack
    !pip install -q -r requirements.txt

Step 5: In the next cell, run the smoke:

    !python smoke_train.py

Step 6: Copy the W&B run URL from the output and retain it with the experiment record.

Run locally (once data/Data_Entry_2017.csv and data/images/ are present)
------------------------------------------------------------------------
PowerShell, session-only credential:

    $env:WANDB_API_KEY = (Read-Host -Prompt "Paste W&B key" -AsSecureString |
        [System.Net.NetworkCredential]::new("", $_).Password)
    .venv\\Scripts\\Activate.ps1
    python smoke_train.py

Rules
-----
- Do not increase MAX_SAMPLES or NUM_EPOCHS without explicit user approval.
- Do not commit WANDB_API_KEY to any file.
- Do not replace the production checkpoint with the smoke checkpoint.
- Record the W&B run URL and exact command output with the experiment evidence.
"""

import os
import sys

# ── Smoke CFG overrides — must happen BEFORE src.* modules are imported ──────
# Python module cache ensures src.train and src.dataset see these values.
from src.config import CFG  # noqa: E402 — import then patch

CFG.MAX_SAMPLES    = 256          # 217 train+val | 39 test after split
CFG.NUM_EPOCHS     = 1            # one epoch; WARMUP_EPOCHS=1 means only the warmup loop runs
CFG.WARMUP_EPOCHS  = 1            # keep <= NUM_EPOCHS so finetune loop is skipped
CFG.NUM_WORKERS    = 2            # Kaggle notebook safe worker count (CPU: 0 also works)
CFG.MODEL_FILENAME = "smoke_densenet121_chestxray.pth"   # isolated output, never overwrites prod

# ── Credential guard ─────────────────────────────────────────────────────────
if not os.environ.get("WANDB_API_KEY"):
    print("ERROR: WANDB_API_KEY is not set.")
    print("")
    print("  Kaggle:     Secrets → Add 'WANDB_API_KEY', toggle 'Attach to notebook' ON.")
    print("  PowerShell: $env:WANDB_API_KEY = (Read-Host -AsSecureString ...)")
    print("              Never write the key to a file or paste it into chat.")
    sys.exit(1)

# ── Run the bounded smoke ─────────────────────────────────────────────────────
from src.train import train  # noqa: E402

if __name__ == "__main__":
    print("=" * 60)
    print("NIH smoke run: 256 samples | 1 epoch | warmup only")
    print(f"  MAX_SAMPLES   = {CFG.MAX_SAMPLES}")
    print(f"  NUM_EPOCHS    = {CFG.NUM_EPOCHS}")
    print(f"  WARMUP_EPOCHS = {CFG.WARMUP_EPOCHS}")
    print(f"  BATCH_SIZE    = {CFG.BATCH_SIZE}")
    print(f"  OUTPUT        = models/{CFG.MODEL_FILENAME}")
    print("=" * 60)

    best_auc = train()

    print("=" * 60)
    print(f"Smoke complete. Best val AUC = {best_auc:.4f}")
    print("Record the W&B run URL with the experiment evidence.")
    print("=" * 60)
