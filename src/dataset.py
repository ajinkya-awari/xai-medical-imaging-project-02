
import os, random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from src.config import CFG


def _build_image_index():
    index = {}
    for folder in CFG.IMAGE_DIRS:
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            if fname.endswith(".png"):
                index[fname] = os.path.join(folder, fname)
    print(f"  Image index built: {len(index):,} images across 12 folders")
    return index


def _load_split_lists():
    with open(CFG.TRAIN_LIST_PATH) as f:
        train_files = set(f.read().splitlines())
    with open(CFG.TEST_LIST_PATH) as f:
        test_files = set(f.read().splitlines())
    return train_files, test_files


def _encode_labels(finding_str):
    findings = finding_str.split("|")
    vec = np.zeros(CFG.NUM_CLASSES, dtype=np.float32)
    for i, disease in enumerate(CFG.DISEASE_LABELS):
        if disease in findings:
            vec[i] = 1.0
    return vec


def get_transforms(split="train"):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((CFG.IMAGE_SIZE + 16, CFG.IMAGE_SIZE + 16)),
            transforms.RandomCrop(CFG.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


class ChestXrayDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records   = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img_path, label = self.records[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)


def build_loaders():
    random.seed(CFG.SEED)
    np.random.seed(CFG.SEED)

    image_index = _build_image_index()
    df = pd.read_csv(CFG.CSV_PATH)
    train_files, test_files = _load_split_lists()

    train_records, test_records = [], []
    for _, row in df.iterrows():
        fname  = row["Image Index"]
        labels = _encode_labels(row["Finding Labels"])
        fpath  = image_index.get(fname)
        if fpath is None:
            continue
        if fname in test_files:
            test_records.append((fpath, labels))
        elif fname in train_files:
            train_records.append((fpath, labels))

    print(f"  Matched {len(train_records):,} train | {len(test_records):,} test images")

    if CFG.MAX_SAMPLES:
        n_train = int(CFG.MAX_SAMPLES * 0.85)
        n_test  = CFG.MAX_SAMPLES - n_train
        random.shuffle(train_records)
        random.shuffle(test_records)
        train_records = train_records[:n_train]
        test_records  = test_records[:n_test]
        print(f"  Subsampled to {len(train_records):,} train+val | {len(test_records):,} test")

    n_val         = int(len(train_records) * CFG.VAL_SPLIT)
    random.shuffle(train_records)
    val_records   = train_records[:n_val]
    train_records = train_records[n_val:]
    print(f"  Final — Train {len(train_records):,} | Val {len(val_records):,} | Test {len(test_records):,}")

    kw = dict(num_workers=CFG.NUM_WORKERS, pin_memory=True)
    train_loader = DataLoader(ChestXrayDataset(train_records, get_transforms("train")), batch_size=CFG.BATCH_SIZE, shuffle=True,  **kw)
    val_loader   = DataLoader(ChestXrayDataset(val_records,   get_transforms("val")),   batch_size=CFG.BATCH_SIZE, shuffle=False, **kw)
    test_loader  = DataLoader(ChestXrayDataset(test_records,  get_transforms("test")),  batch_size=CFG.BATCH_SIZE, shuffle=False, **kw)

    return train_loader, val_loader, test_loader, test_records
