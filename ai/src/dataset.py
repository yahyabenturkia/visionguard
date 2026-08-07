"""
VisionGuard — Dataset & Preprocessing
=====================================
The correctness-critical module. Two responsibilities:

1. preprocess_crop(): the ONE preprocessing function (grayscale + CLAHE +
   resize + normalize). Imported by BOTH training and live inference so the
   model always sees identically-processed input. Never duplicate this logic
   elsewhere.

2. Leakage-safe grouping: crops sharing a source image are kept together when
   splitting, so near-duplicate overlapping crops never straddle
   train/val/test (which would inflate validation scores).

The label of a crop is determined by the FOLDER it lives in (good/ or defect/),
NOT by its filename — filenames carry source provenance only.
"""

import os
import re
import glob
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import config as C


# ─── SOURCE-IMAGE GROUPING KEY (leakage prevention) ───────────────────────────
_V_SUFFIX = re.compile(r"_v\d+$")

def source_id(filename):
    """
    Map a crop filename to its source-image id by stripping the _v{N} suffix
    and the extension. Both crops of one source image share this id, so they
    are forced into the same split.

      defect_defect_L_20260608_191627_233315_v1.jpg
      defect_defect_L_20260608_191627_233315_v2.jpg
        -> 'defect_defect_L_20260608_191627_233315'
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    return _V_SUFFIX.sub("", stem)


# ─── PREPROCESSING (shared by training AND inference) ─────────────────────────
_clahe = cv2.createCLAHE(clipLimit=C.CLAHE_CLIP, tileGridSize=C.CLAHE_GRID)

def preprocess_crop(img_bgr):
    """
    Turn a raw BGR crop (any size) into a normalized CHW float tensor ready
    for the model. This EXACT function runs at training and at inference.

    Steps: grayscale -> CLAHE -> resize -> replicate to 3ch -> ImageNet-normalize.
    Returns: torch.FloatTensor of shape (3, INPUT_SIZE, INPUT_SIZE).
    """
    if C.USE_GRAYSCALE:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  # base is gray regardless

    if C.USE_CLAHE:
        gray = _clahe.apply(gray)

    gray = cv2.resize(gray, (C.INPUT_SIZE, C.INPUT_SIZE), interpolation=cv2.INTER_AREA)

    # replicate gray -> 3 channels so ImageNet-pretrained backbone works as-is
    img = np.stack([gray, gray, gray], axis=-1).astype(np.float32) / 255.0

    # ImageNet normalization
    mean = np.array(C.NORM_MEAN, dtype=np.float32)
    std  = np.array(C.NORM_STD, dtype=np.float32)
    img = (img - mean) / std

    # HWC -> CHW
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
    return tensor


# ─── AUGMENTATION (train only) ────────────────────────────────────────────────
def augment_bgr(img_bgr):
    """
    Light geometric/photometric augmentation applied to the raw BGR crop
    BEFORE preprocess_crop(). Train split only. Keeps the defect signal intact
    (gentle rotation/flip/brightness — nothing that erases the shadow).
    """
    h, w = img_bgr.shape[:2]

    # horizontal flip
    if C.AUG_HFLIP and random.random() < 0.5:
        img_bgr = cv2.flip(img_bgr, 1)

    # rotation +/- AUG_ROTATION_DEG
    if C.AUG_ROTATION_DEG > 0:
        ang = random.uniform(-C.AUG_ROTATION_DEG, C.AUG_ROTATION_DEG)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        img_bgr = cv2.warpAffine(img_bgr, M, (w, h),
                                 borderMode=cv2.BORDER_REFLECT_101)

    # brightness / contrast jitter
    if C.AUG_BRIGHTNESS > 0 or C.AUG_CONTRAST > 0:
        alpha = 1.0 + random.uniform(-C.AUG_CONTRAST, C.AUG_CONTRAST)   # contrast
        beta  = 255.0 * random.uniform(-C.AUG_BRIGHTNESS, C.AUG_BRIGHTNESS)  # brightness
        img_bgr = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)

    return img_bgr


# ─── TORCH DATASET ────────────────────────────────────────────────────────────
class VeinDataset(Dataset):
    """
    Reads (filepath, label) pairs. label from folder: good=0, defect=1.
    Applies augmentation only when train=True, then the shared preprocess.
    """
    def __init__(self, samples, train=False):
        self.samples = samples       # list of (path, label)
        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"could not read {path}")
        if self.train:
            img = augment_bgr(img)
        tensor = preprocess_crop(img)
        return tensor, label


# ─── LEAKAGE-SAFE SPLIT ───────────────────────────────────────────────────────
def gather_samples(crops_dir):
    """Collect (path, label) for every crop, label by folder."""
    samples = []
    for label_idx, cls in enumerate(C.CLASSES):       # good=0, defect=1
        folder = os.path.join(crops_dir, cls)
        for f in glob.glob(os.path.join(folder, "*.jpg")) + \
                 glob.glob(os.path.join(folder, "*.png")):
            samples.append((f, label_idx))
    return samples


def split_by_source(samples, split=None, seed=None):
    """
    Group samples by source image, then assign whole groups to train/val/test
    so no source image's crops are split across sets.

    Returns dict: {"train": [...], "val": [...], "test": [...]}.
    """
    split = split or C.SPLIT
    seed = C.SPLIT_SEED if seed is None else seed

    # group sample indices by source id
    groups = {}
    for s in samples:
        sid = source_id(s[0])
        groups.setdefault(sid, []).append(s)

    group_ids = sorted(groups.keys())     # sorted for determinism
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    n = len(group_ids)
    n_train = int(round(n * split["train"]))
    n_val   = int(round(n * split["val"]))
    train_ids = group_ids[:n_train]
    val_ids   = group_ids[n_train:n_train + n_val]
    test_ids  = group_ids[n_train + n_val:]

    out = {"train": [], "val": [], "test": []}
    for gid in train_ids:
        out["train"].extend(groups[gid])
    for gid in val_ids:
        out["val"].extend(groups[gid])
    for gid in test_ids:
        out["test"].extend(groups[gid])
    return out


def class_weights(samples):
    """Inverse-frequency weights for the loss (mild; classes ~balanced)."""
    counts = [0] * C.NUM_CLASSES
    for _, lbl in samples:
        counts[lbl] += 1
    total = sum(counts)
    weights = [total / (C.NUM_CLASSES * max(1, c)) for c in counts]
    return torch.tensor(weights, dtype=torch.float32), counts