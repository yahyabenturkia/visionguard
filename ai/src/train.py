"""
VisionGuard — Training (modular)
================================
ResNet18 transfer-learning trainer. Imports config.py and dataset.py so the
SAME preprocessing/config is shared with inference — no duplicated logic.

Runs identically locally (CPU/GPU) and on Kaggle (GPU). On Kaggle, set the
dataset/model paths via env vars BEFORE importing config:

    import os
    os.environ["VG_DATASET_DIR"] = "/kaggle/input/.../dataset"
    os.environ["VG_MODELS_DIR"]  = "/kaggle/working/models"
    import train; train.main()

Two-phase transfer learning:
  Phase 1 — freeze backbone, train classifier head.
  Phase 2 — unfreeze last block (layer4 + fc), fine-tune at low LR.

Outputs the best model (by val loss) to MODELS_DIR and prints test metrics
(accuracy + defect precision/recall/F1 + confusion matrix).
"""

import os
import time
import copy
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

import config as C
from dataset import VeinDataset, gather_samples, split_by_source, class_weights


# ─── reproducibility ──────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── data ─────────────────────────────────────────────────────────────────────
def build_loaders():
    """
    Build train/val/test loaders.

    Two supported layouts (auto-detected):
      A) DATASET_DIR has train/ val/ test/ subfolders (from preprocess.py).
         -> load each split folder directly.
      B) DATASET_DIR has good/ defect/ only (raw crops).
         -> split in-memory by source image (leakage-safe).
    """
    has_split = all(os.path.isdir(os.path.join(C.DATASET_DIR, s))
                    for s in ("train", "val", "test"))

    if has_split:
        print("  detected pre-split dataset (train/val/test folders)")
        train_ds = _folder_dataset("train", train=True)
        val_ds   = _folder_dataset("val",   train=False)
        test_ds  = _folder_dataset("test",  train=False)
    else:
        print("  no split folders found -> splitting in-memory by source image")
        samples = gather_samples(C.DATASET_DIR)
        sp = split_by_source(samples)
        train_ds = VeinDataset(sp["train"], train=True)
        val_ds   = VeinDataset(sp["val"],   train=False)
        test_ds  = VeinDataset(sp["test"],  train=False)

    print(f"  sizes -> train {len(train_ds)}  val {len(val_ds)}  test {len(test_ds)}")

    train_ld = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True,
                          num_workers=C.NUM_WORKERS, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=C.BATCH_SIZE, shuffle=False,
                          num_workers=C.NUM_WORKERS, pin_memory=True)
    test_ld  = DataLoader(test_ds,  batch_size=C.BATCH_SIZE, shuffle=False,
                          num_workers=C.NUM_WORKERS, pin_memory=True)
    return train_ds, train_ld, val_ld, test_ld


def _folder_dataset(split, train):
    """Build a VeinDataset from DATASET_DIR/<split>/<class>/ folders."""
    import glob
    samples = []
    for label, cls in enumerate(C.CLASSES):
        folder = os.path.join(C.DATASET_DIR, split, cls)
        for f in glob.glob(os.path.join(folder, "*.jpg")) + \
                 glob.glob(os.path.join(folder, "*.png")):
            samples.append((f, label))
    return VeinDataset(samples, train=train)


# ─── model ────────────────────────────────────────────────────────────────────
def build_model(device):
    if C.MODEL_NAME == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1
                            if C.PRETRAINED else None)
        m.fc = nn.Linear(m.fc.in_features, C.NUM_CLASSES)
    elif C.MODEL_NAME == "mobilenet_v2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
                                if C.PRETRAINED else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, C.NUM_CLASSES)
    else:
        raise ValueError(f"unknown MODEL_NAME: {C.MODEL_NAME}")
    return m.to(device)


def set_head_only(model):
    """Phase 1: freeze everything except the classifier head."""
    for name, p in model.named_parameters():
        p.requires_grad = _is_head(name)

def unfreeze_last_block(model):
    """Phase 2: train head + last block."""
    for name, p in model.named_parameters():
        p.requires_grad = _is_head(name) or _is_last_block(name)

def _is_head(name):
    return name.startswith("fc.") or name.startswith("classifier.")

def _is_last_block(name):
    # resnet18 last block = layer4; mobilenet_v2 last conv = features.18
    return name.startswith("layer4.") or name.startswith("features.18.")


# ─── train / eval ─────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if train:
                loss.backward(); optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            tot += x.size(0)
    return loss_sum / tot, correct / tot


def train_phase(model, train_ld, val_ld, criterion, device,
                epochs, lr, phase, patience):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=C.WEIGHT_DECAY)
    best_val, best_state, wait = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_ld, criterion, device, opt)
        va_loss, va_acc = run_epoch(model, val_ld, criterion, device)
        print(f"[{phase}] epoch {ep:2d}/{epochs} | "
              f"train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f} | {time.time()-t0:.0f}s")
        if va_loss < best_val:
            best_val, best_state, wait = va_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  early stop (no val improvement for {patience} epochs)")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val


def evaluate_test(model, test_ld, device):
    model.eval()
    preds, trues = [], []
    probs = []
    with torch.no_grad():
        for x, y in test_ld:
            out = model(x.to(device))
            p = torch.softmax(out, dim=1)[:, 1]   # P(defect) for each sample
            preds += out.argmax(1).cpu().tolist()
            probs += p.cpu().tolist()
            trues += y.tolist()
    tp = sum(1 for p, t in zip(preds, trues) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(preds, trues) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(preds, trues) if p == 0 and t == 1)
    tn = sum(1 for p, t in zip(preds, trues) if p == 0 and t == 0)
    n = max(1, len(trues))
    acc  = (tp + tn) / n
    prec = tp / max(1, tp + fp)
    rec  = tp / max(1, tp + fn)
    f1   = 2 * prec * rec / max(1e-9, prec + rec)
    print("\n===== TEST RESULTS =====")
    print(f" samples: {len(trues)}  (true dist: {Counter(trues)})")
    print(f" confusion:  TN {tn}  FP {fp}  FN {fn}  TP {tp}")
    print(f" accuracy : {acc*100:.1f}%")
    print(f" DEFECT precision: {prec*100:.1f}%")
    print(f" DEFECT recall   : {rec*100:.1f}%   <-- key metric (missed defects)")
    print(f" DEFECT F1       : {f1*100:.1f}%")
    # per-class accuracy
    good_total = sum(1 for t in trues if t == 0)
    def_total  = sum(1 for t in trues if t == 1)
    good_acc = tn / max(1, good_total)
    def_acc  = tp / max(1, def_total)
    print(f" per-class accuracy: good {good_acc*100:.1f}%  defect {def_acc*100:.1f}%")

    # ROC-AUC (needs sklearn)
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(trues, probs)
        print(f" ROC-AUC: {auc:.3f}  (1.0 = perfect, 0.5 = random)")
    except Exception as e:
        print(f" ROC-AUC: skipped ({e})")
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    set_seed(C.SPLIT_SEED)
    device = "cuda" if (C.DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
    os.makedirs(C.MODELS_DIR, exist_ok=True)

    print("=" * 56)
    print(f"  VisionGuard Training — {C.MODEL_NAME}")
    print(f"  device: {device}  | dataset: {C.DATASET_DIR}")
    print("=" * 56)

    train_ds, train_ld, val_ld, test_ld = build_loaders()

    # class weights from the train split
    cw, counts = class_weights(train_ds.samples)
    cw = cw.to(device)
    print(f"  train class counts {counts}  weights {[round(x,3) for x in cw.tolist()]}")
    criterion = nn.CrossEntropyLoss(weight=cw if C.USE_CLASS_WEIGHTS else None)

    model = build_model(device)

    print("\n=== PHASE 1: train head (backbone frozen) ===")
    set_head_only(model)
    train_phase(model, train_ld, val_ld, criterion, device,
                C.EPOCHS_HEAD, C.LR_HEAD, "HEAD", C.EARLY_STOP_PATIENCE)

    if C.FINETUNE_LAST_BLOCK:
        print("\n=== PHASE 2: fine-tune last block ===")
        unfreeze_last_block(model)
        train_phase(model, train_ld, val_ld, criterion, device,
                    C.EPOCHS_FINETUNE, C.LR_FINETUNE, "FT", C.EARLY_STOP_PATIENCE)

    best_path = os.path.join(C.MODELS_DIR, C.BEST_MODEL_NAME)
    torch.save(model.state_dict(), best_path)
    print(f"\nsaved model: {best_path}")

    evaluate_test(model, test_ld, device)


if __name__ == "__main__":
    main()