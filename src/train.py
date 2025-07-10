
import os, time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from src.config import CFG
from src.model import ChestXrayModel
from src.dataset import build_loaders


def _auc_score(labels, preds):
    aucs = []
    for i in range(labels.shape[1]):
        if len(np.unique(labels[:, i])) > 1:
            aucs.append(roc_auc_score(labels[:, i], preds[:, i]))
    return float(np.mean(aucs)) if aucs else 0.5


def _run_epoch(model, loader, criterion, optimizer, device, training=True):
    model.train(training)
    total_loss, all_labels, all_preds = 0.0, [], []
    with torch.set_grad_enabled(training):
        for images, labels in tqdm(loader, leave=False, desc="  batches"):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss   = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
    return total_loss / len(loader.dataset), _auc_score(np.vstack(all_labels), np.vstack(all_preds))


def train():
    torch.manual_seed(CFG.SEED)
    np.random.seed(CFG.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    train_loader, val_loader, _, _ = build_loaders()
    model     = ChestXrayModel(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    os.makedirs(CFG.MODEL_DIR, exist_ok=True)
    best_auc, patience, best_epoch = 0.0, 0, 0

    print(f"\n  [Phase 1] Warm-up: head only for {CFG.WARMUP_EPOCHS} epochs")
    model.freeze_backbone()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=CFG.LR_HEAD, weight_decay=CFG.WEIGHT_DECAY)
    for epoch in range(1, CFG.WARMUP_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_auc = _run_epoch(model, train_loader, criterion, optimizer, device, True)
        va_loss, va_auc = _run_epoch(model, val_loader,   criterion, optimizer, device, False)
        print(f"  Epoch {epoch:02d}/{CFG.NUM_EPOCHS} [warm-up]   train loss={tr_loss:.4f} AUC={tr_auc:.4f} | val loss={va_loss:.4f} AUC={va_auc:.4f} ({time.time()-t0:.0f}s)")

    print(f"\n  [Phase 2] Fine-tuning full network for {CFG.NUM_EPOCHS - CFG.WARMUP_EPOCHS} epochs")
    model.unfreeze_backbone()
    optimizer = Adam(model.parameters(), lr=CFG.LR_FINETUNE, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    for epoch in range(CFG.WARMUP_EPOCHS + 1, CFG.NUM_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_auc = _run_epoch(model, train_loader, criterion, optimizer, device, True)
        va_loss, va_auc = _run_epoch(model, val_loader,   criterion, optimizer, device, False)
        scheduler.step(va_auc)
        print(f"  Epoch {epoch:02d}/{CFG.NUM_EPOCHS} [fine-tune] train loss={tr_loss:.4f} AUC={tr_auc:.4f} | val loss={va_loss:.4f} AUC={va_auc:.4f} ({time.time()-t0:.0f}s)")

        if va_auc > best_auc:
            best_auc, best_epoch, patience = va_auc, epoch, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_auc": va_auc, "val_loss": va_loss},
                       os.path.join(CFG.MODEL_DIR, CFG.MODEL_FILENAME))
            print(f"    Saved best model (val AUC={best_auc:.4f})")
        else:
            patience += 1
            if patience >= CFG.EARLY_STOP_PAT:
                print(f"    Early stopping at epoch {epoch}")
                break

    print(f"\n  Training done. Best val AUC={best_auc:.4f} at epoch {best_epoch}")
    return best_auc
