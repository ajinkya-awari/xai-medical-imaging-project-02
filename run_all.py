"""
Master script — runs the full XAI chest X-ray pipeline end-to-end.

Executes in order:
  1. Train DenseNet121 on chest X-ray data
  2. Evaluate on held-out test set
  3. Generate Grad-CAM sample visualisations

After this finishes, you can launch the Streamlit app:
    streamlit run app.py

Usage:
    python run_all.py
"""

import time
import sys
from pathlib import Path

# make sure we can import from src/
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    start = time.time()
    print("\n" + "=" * 60)
    print("  XAI Chest X-ray Classification Pipeline")
    print("=" * 60)

    # ── Step 1: Training ──────────────────────────────────────────
    print("\n[1/3] Training DenseNet121...")
    print("-" * 40)
    from src.train import train
    train()

    # ── Step 2: Evaluation ────────────────────────────────────────
    print("\n[2/3] Evaluating on test set...")
    print("-" * 40)
    from src.evaluate import evaluate
    evaluate()

    # ── Step 3: Grad-CAM visualisations ───────────────────────────
    print("\n[3/3] Generating Grad-CAM samples...")
    print("-" * 40)
    from src.visualize import generate_samples
    generate_samples(n_samples=6)

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete in {elapsed/60:.1f} minutes")
    print(f"  Outputs saved to: outputs/")
    print(f"  Model saved to:   models/densenet121_chestxray.pth")
    print(f"\n  To launch the web app:")
    print(f"    streamlit run app.py")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
