"""
VisionGuard — AI Pipeline Configuration
=======================================
Single source of truth for every hyperparameter and path used by
preprocess.py, dataset.py, train.py, and evaluate.py.

Change a value here and it propagates everywhere — keeps local and
Kaggle runs consistent and makes the report's "implementation details"
section trivial to fill in.
"""

import os

# ─── PATHS ────────────────────────────────────────────────────────────────────
# Project root is the parent of this file's folder (ai/src/ -> ai/)
AI_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CROPS_DIR   = os.path.join(AI_DIR, "crops")      # raw labeled crops (good/ defect/)
DATASET_DIR = os.path.join(AI_DIR, "dataset")    # processed + split output
MODELS_DIR  = os.path.join(AI_DIR, "models")     # trained weights land here

CLASSES = ["good", "defect"]                     # index 0 = good, 1 = defect

# On Kaggle the dataset lives elsewhere; override with an env var if set.
# e.g. export VG_DATASET_DIR=/kaggle/input/visionguard-dataset
DATASET_DIR = os.environ.get("VG_DATASET_DIR", DATASET_DIR)
MODELS_DIR  = os.environ.get("VG_MODELS_DIR", MODELS_DIR)

# ─── SPLIT ────────────────────────────────────────────────────────────────────
SPLIT = {"train": 0.80, "val": 0.10, "test": 0.10}
SPLIT_SEED = 42                                  # reproducible splits
# CRITICAL: crops sharing a source image must stay in the SAME split.
# Grouping key is derived from the crop filename (see dataset.py).

# ─── PREPROCESSING ────────────────────────────────────────────────────────────
# These define how a raw crop becomes a model input. Used IDENTICALLY at
# training time and at live inference time — never change one without the other.
INPUT_SIZE   = 224          # ResNet18 native input (square)
USE_GRAYSCALE = True        # convert to grayscale (validated: removes color nuisance)
USE_CLAHE     = True        # local contrast enhancement (validated: lifts shadow signal)
CLAHE_CLIP    = 2.0         # CLAHE clip limit (gentle; higher = more aggressive)
CLAHE_GRID    = (8, 8)      # CLAHE tile grid

# Grayscale is replicated to 3 channels so the ImageNet-pretrained ResNet
# (which expects 3-channel input) can be used without surgery.
# ImageNet normalization stats (applied after the gray->3ch replication):
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

# ─── AUGMENTATION (train only) ────────────────────────────────────────────────
AUG_ROTATION_DEG   = 15     # +/- degrees — handles vein tilt variation
AUG_HFLIP          = True   # horizontal flip
AUG_BRIGHTNESS     = 0.15   # brightness jitter — robustness to lighting
AUG_CONTRAST       = 0.15   # contrast jitter

# ─── MODEL ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "resnet18"   # primary choice; "mobilenet_v2" is the fallback
NUM_CLASSES  = 2
PRETRAINED   = True         # ImageNet weights (transfer learning)
# ─── INFERENCE (confidence thresholds) ────────────────────────────────────────
# The model outputs P(defect) per vein. These thresholds turn that probability
# into a 3-way decision so borderline veins are flagged for review rather than
# guessed. Tunable after observing real live behavior.
#   P(defect) >= DEFECT_THRESH         -> "defect"
#   P(defect) <= GOOD_THRESH           -> "good"
#   in between                         -> "uncertain" (human review)
DEFECT_THRESH = 0.80     # >= this P(defect) => defect
GOOD_THRESH   = 0.20     # <= this P(defect) => good
# (between 0.20 and 0.80 => uncertain)
# ─── CROP & DETECTION (shared with labeler + inference) ───────────────────────
CROP_W = 140          # crop width  (px) — MUST match training crops
CROP_H = 60           # crop height (px) — MUST match training crops

DET_THRESH   = 200    # brightness threshold to isolate lit veins
DET_AREA_MIN = 150    # min blob area (px)
DET_AREA_MAX = 6000   # max blob area
DET_AR_MIN   = 1.6    # min aspect ratio (elongated slots)
DET_MORPH_K  = 5      # morphological kernel size
# Path to the trained model used for inference
INFER_MODEL_PATH = os.path.join(MODELS_DIR, "resnet18_v2.pth")
# Transfer-learning strategy: freeze backbone first, then fine-tune last block.
FREEZE_BACKBONE      = True   # phase 1: train classifier head only
FINETUNE_LAST_BLOCK  = True   # phase 2: unfreeze last block at low LR

# ─── TRAINING ─────────────────────────────────────────────────────────────────
BATCH_SIZE       = 32
EPOCHS_HEAD      = 12        # phase 1: head-only epochs
EPOCHS_FINETUNE  = 8         # phase 2: fine-tuning epochs
LR_HEAD          = 1e-3      # learning rate for the head
LR_FINETUNE      = 1e-5      # low LR for fine-tuning (avoid wrecking pretrained features)
WEIGHT_DECAY     = 1e-4
EARLY_STOP_PATIENCE = 5      # stop if val loss doesn't improve for N epochs

# Class weighting in the loss — mild safeguard even though classes are ~balanced.
USE_CLASS_WEIGHTS = True

NUM_WORKERS = 2             # dataloader workers (Kaggle handles 2 fine)
DEVICE      = "cuda"        # train.py falls back to cpu if cuda unavailable

# ─── OUTPUT ───────────────────────────────────────────────────────────────────
BEST_MODEL_NAME = f"{MODEL_NAME}_best.pth"
LAST_MODEL_NAME = f"{MODEL_NAME}_last.pth"