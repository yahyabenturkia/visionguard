"""
VisionGuard — Local Dataset Preparation (torch-free)
====================================================
Run this LOCALLY. It takes the raw labeled crops and writes a leakage-safe
80/10/10 split into dataset/{train,val,test}/{good,defect}/, ready to zip and
upload to Kaggle.

  python3 preprocess.py

Design notes
  - RAW crops are copied (not pre-processed). Grayscale+CLAHE and augmentation
    are applied at TRAINING load time (dataset.py), because augmentation needs
    the raw image. This keeps the uploaded dataset flexible and reproducible.
  - The split groups crops by SOURCE IMAGE (the _v{N} suffix is stripped), so
    overlapping crops from one frame never straddle splits. This algorithm is
    identical to the one verified in dataset.py.
  - Requires only: opencv-python (for a quick size/readability check), numpy.
    No PyTorch needed locally.

Outputs
  - dataset/train|val|test/good|defect/  (the split image files)
  - dataset/split_manifest.csv           (filepath,source_id,class,split)
"""

import os
import re
import csv
import glob
import random
import shutil
import config as C

try:
    import cv2
    _HAVE_CV2 = True
except ImportError:
    _HAVE_CV2 = False


# ─── grouping key (matches dataset.py) ────────────────────────────────────────
_V_SUFFIX = re.compile(r"_v\d+$")

def source_id(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return _V_SUFFIX.sub("", stem)


# ─── gather ───────────────────────────────────────────────────────────────────
def gather_samples(crops_dir):
    samples = []   # (path, class_name)
    for cls in C.CLASSES:
        folder = os.path.join(crops_dir, cls)
        if not os.path.isdir(folder):
            print(f"  WARNING: missing folder {folder}")
            continue
        for f in glob.glob(os.path.join(folder, "*.jpg")) + \
                 glob.glob(os.path.join(folder, "*.png")):
            samples.append((f, cls))
    return samples


# ─── leakage-safe split (matches dataset.py algorithm) ────────────────────────
def split_by_source(samples, split, seed):
    groups = {}
    for s in samples:
        groups.setdefault(source_id(s[0]), []).append(s)

    group_ids = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    n = len(group_ids)
    n_train = int(round(n * split["train"]))
    n_val   = int(round(n * split["val"]))
    train_ids = group_ids[:n_train]
    val_ids   = group_ids[n_train:n_train + n_val]
    test_ids  = group_ids[n_train + n_val:]

    out = {"train": [], "val": [], "test": []}
    for gid in train_ids: out["train"].extend(groups[gid])
    for gid in val_ids:   out["val"].extend(groups[gid])
    for gid in test_ids:  out["test"].extend(groups[gid])
    return out


# ─── size/readability sanity check ────────────────────────────────────────────
def sanity_check(samples):
    if not _HAVE_CV2:
        print("  (cv2 not available — skipping size check)")
        return True
    sizes = {}
    bad = []
    for path, _ in samples:
        img = cv2.imread(path)
        if img is None:
            bad.append(path); continue
        sizes[(img.shape[1], img.shape[0])] = sizes.get((img.shape[1], img.shape[0]), 0) + 1
    print(f"  crop sizes present: {sizes}")
    if len(sizes) > 1:
        print("  WARNING: multiple crop sizes detected — model expects uniform size.")
    if bad:
        print(f"  WARNING: {len(bad)} unreadable files")
    return len(bad) == 0


# ─── write split to disk ──────────────────────────────────────────────────────
def write_split(split_map):
    # fresh output tree
    if os.path.isdir(C.DATASET_DIR):
        print(f"  removing existing {C.DATASET_DIR}")
        shutil.rmtree(C.DATASET_DIR)
    for sp in ("train", "val", "test"):
        for cls in C.CLASSES:
            os.makedirs(os.path.join(C.DATASET_DIR, sp, cls), exist_ok=True)

    manifest_rows = []
    for sp, items in split_map.items():
        for path, cls in items:
            dst = os.path.join(C.DATASET_DIR, sp, cls, os.path.basename(path))
            shutil.copy2(path, dst)
            manifest_rows.append([os.path.basename(path), source_id(path), cls, sp])

    # manifest CSV
    man_path = os.path.join(C.DATASET_DIR, "split_manifest.csv")
    with open(man_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "source_id", "class", "split"])
        w.writerows(manifest_rows)
    return man_path


# ─── leakage verification ─────────────────────────────────────────────────────
def verify_no_leakage(split_map):
    sids = {sp: set(source_id(p) for p, _ in items) for sp, items in split_map.items()}
    ok = True
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        inter = sids[a] & sids[b]
        if inter:
            ok = False
            print(f"  LEAKAGE ERROR: {a} ∩ {b} = {len(inter)} shared source images!")
    if ok:
        print("  leakage check: PASSED (0 shared source images across splits)")
    return ok


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 56)
    print("  VisionGuard — Dataset Preparation")
    print("=" * 56)
    print(f"  crops in : {C.CROPS_DIR}")
    print(f"  output   : {C.DATASET_DIR}")
    print(f"  split    : {C.SPLIT}  seed {C.SPLIT_SEED}")

    if not os.path.isdir(C.CROPS_DIR):
        print(f"\n  ERROR: crops folder not found: {C.CROPS_DIR}")
        return

    samples = gather_samples(C.CROPS_DIR)
    counts = {cls: sum(1 for _, c in samples if c == cls) for cls in C.CLASSES}
    print(f"\n  total crops: {len(samples)}  ->  {counts}")
    if len(samples) == 0:
        print("  ERROR: no crops found."); return

    print("\n  sanity check:")
    sanity_check(samples)

    n_sources = len(set(source_id(p) for p, _ in samples))
    print(f"\n  unique source images: {n_sources}")

    split_map = split_by_source(samples, C.SPLIT, C.SPLIT_SEED)
    print("\n  split result:")
    for sp in ("train", "val", "test"):
        c = {cls: sum(1 for _, cl in split_map[sp] if cl == cls) for cls in C.CLASSES}
        print(f"    {sp:5s}: {len(split_map[sp]):4d}  {c}")

    print()
    if not verify_no_leakage(split_map):
        print("\n  ABORTING: leakage detected, not writing dataset.")
        return

    print("\n  writing split to disk...")
    man = write_split(split_map)
    print(f"  done. manifest: {man}")
    print("\n  Next: zip the 'dataset' folder and upload to Kaggle.")
    print("=" * 56)


if __name__ == "__main__":
    main()