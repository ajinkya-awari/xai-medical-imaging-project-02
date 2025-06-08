import os

class CFG:
    # ── Dataset paths ──────────────────────────────────────────────────────────
    # For LOCAL use: set DATA_DIR to where you extracted the NIH dataset.
    # For KAGGLE:    DATA_DIR is set automatically via environment detection.

    _KAGGLE = os.path.exists("/kaggle/input/datasets/organizations/nih-chest-xrays/data")

    if _KAGGLE:
        DATA_DIR        = "/kaggle/input/datasets/organizations/nih-chest-xrays/data"
    else:
        # ── LOCAL PATH — change this to your NIH dataset folder ──────────────
        DATA_DIR        = "data"   # e.g. "C:/datasets/nih-chestxray" or "/home/user/nih"

    CSV_PATH        = os.path.join(DATA_DIR, "Data_Entry_2017.csv")
    TRAIN_LIST_PATH = os.path.join(DATA_DIR, "train_val_list.txt")
    TEST_LIST_PATH  = os.path.join(DATA_DIR, "test_list.txt")

    # For Kaggle the images are split across 12 subfolders.
    # For local use they are typically all in data/images/
    if _KAGGLE:
        IMAGE_DIRS = [
            os.path.join(DATA_DIR, f"images_{str(i).zfill(3)}", "images")
            for i in range(1, 13)
        ]
    else:
        IMAGE_DIRS = [os.path.join(DATA_DIR, "images")]

    # ── Output paths ───────────────────────────────────────────────────────────
    MODEL_DIR       = "models"
    OUTPUT_DIR      = "outputs"
    MODEL_FILENAME  = "densenet121_chestxray.pth"

    # ── Data settings ──────────────────────────────────────────────────────────
    # Set MAX_SAMPLES = None to use the full 112,120-image dataset
    MAX_SAMPLES     = 20000
    IMAGE_SIZE      = 224
    VAL_SPLIT       = 0.15
    NUM_WORKERS     = 4

    # ── Training settings ──────────────────────────────────────────────────────
    BATCH_SIZE      = 64
    NUM_EPOCHS      = 10
    WARMUP_EPOCHS   = 2
    LR_HEAD         = 1e-3
    LR_FINETUNE     = 1e-4
    WEIGHT_DECAY    = 1e-5
    DROPOUT         = 0.3
    EARLY_STOP_PAT  = 3
    SEED            = 42
    NUM_CLASSES     = 14

    # ── Disease labels (NIH order — DO NOT REORDER) ────────────────────────────
    DISEASE_LABELS  = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
        "Mass", "Nodule", "Pneumonia", "Pneumothorax",
        "Consolidation", "Edema", "Emphysema", "Fibrosis",
        "Pleural_Thickening", "Hernia",
    ]
