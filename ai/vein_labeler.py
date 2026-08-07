#!/usr/bin/env python3
"""
VisionGuard — Semi-Automatic Vein Labeling Tool
================================================
Loads full-frame M00 images from dataset/good and dataset/defect, auto-detects
lit veins, draws boxes you classify with a click, then crops each labeled vein
and sorts it into crops/good or crops/defect.

NOTE: the dataset good/defect folders are IMAGE-level labels and are IGNORED for
crop labels. You judge each vein yourself; a "defect" full frame still contains
many good veins.

WORKFLOW PER IMAGE
  - Auto-detected veins appear as GREY boxes.
  - Left-click a box  : cycle  GREY -> GREEN (good) -> RED (defect) -> GREY.
  - Left-click empty  : add a new box there (for veins detection missed).
  - Hover + press 'd' : delete the box under the cursor (reliable on all backends).
  - Right-click a box : also deletes (may be intercepted by Qt; use 'd' if so).
  - GREY boxes are NOT saved. Only GREEN/RED are cropped and sorted.

KEYS
  n / SPACE / ->  : save labeled crops, go to NEXT image
  p / <-          : previous image (re-label; overwrites that image's crops)
  + / -           : crop WIDTH  bigger / smaller
  w / s           : crop HEIGHT bigger / smaller
  d               : delete box under cursor
  r               : reset boxes on this image (re-run auto-detect)
  q / ESC         : save current image, then quit (resume later from here)

RESUME
  Processed images are recorded in the manifest file (see MANIFEST below).
  Re-running skips already-done images automatically.

SETUP
  pip install opencv-python numpy   (NOT the headless variant)
"""

import cv2
import numpy as np
import os
import sys

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
DATASET_DIR = os.path.expanduser("~/visionguard/raw_data")        # has good/ and defect/ subfolders of FULL-FRAME images
SRC_SUBDIRS = ["good", "defect"]                                 # read full frames from both
OUTPUT_GOOD = os.path.expanduser("~/visionguard/ai/crops/good")  # vein crops out
OUTPUT_DEF  = os.path.expanduser("~/visionguard/ai/crops/defect")
MANIFEST    = os.path.expanduser("~/visionguard/ai/labeled_manifest.txt")

CROP_W      = 110        # crop width  (px) — adjust live with + / -
CROP_H      = 110        # crop height (px) — adjust live with w / s
CROP_MIN    = 60
CROP_MAX    = 140
CROP_STEP   = 10

# Vein auto-detection parameters (tuned on real VisionGuard images)
DET_THRESH   = 200       # brightness threshold to isolate lit veins
DET_AREA_MIN = 150       # min blob area (px) to count as a vein
DET_AREA_MAX = 6000      # max blob area
DET_AR_MIN   = 1.6       # min aspect ratio (veins are elongated slots)
DET_MORPH_K  = 5         # morphological kernel size for cleanup

# Display
MAX_DISPLAY_W = 1280     # max window width
MAX_DISPLAY_H = 720      # max window height (incl. header) — lower if still too tall
HEADER_H      = 70
WIN = "VisionGuard Vein Labeler"

# Box states
GREY, GOOD, DEFECT = 0, 1, 2
STATE_COLOR = {GREY: (180,180,180), GOOD: (0,200,0), DEFECT: (0,0,255)}

# ─── DETECTION ────────────────────────────────────────────────────────────────
def detect_veins(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, DET_THRESH, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DET_MORPH_K, DET_MORPH_K))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < DET_AREA_MIN or a > DET_AREA_MAX:
            continue
        x, y, w, h = cv2.boundingRect(c)
        ar = max(w, h) / max(1, min(w, h))
        if ar < DET_AR_MIN:
            continue
        centers.append((x + w // 2, y + h // 2))
    centers.sort(key=lambda p: p[0])
    return centers

# ─── CROP + SAVE ──────────────────────────────────────────────────────────────
def crop_at(img, cx, cy, w, h):
    H, W = img.shape[:2]
    hw, hh = w // 2, h // 2
    x1, y1 = max(0, cx - hw), max(0, cy - hh)
    x2, y2 = min(W, cx + hw), min(H, cy + hh)
    crop = img[y1:y2, x1:x2]
    pw, ph = w - (x2 - x1), h - (y2 - y1)
    if pw > 0 or ph > 0:
        crop = cv2.copyMakeBorder(crop, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(0,0,0))
    return crop

def save_crops(img, boxes, stem, w, h):
    saved = {"good": 0, "defect": 0}
    idx = 0
    for b in boxes:
        if b["state"] == GOOD:
            d = OUTPUT_GOOD; key = "good"
        elif b["state"] == DEFECT:
            d = OUTPUT_DEF;  key = "defect"
        else:
            continue
        idx += 1
        crop = crop_at(img, b["cx"], b["cy"], w, h)
        cv2.imwrite(os.path.join(d, f"{stem}_v{idx}.jpg"), crop)
        saved[key] += 1
    return saved

# ─── MANIFEST ─────────────────────────────────────────────────────────────────
def load_manifest(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return set(line.strip() for line in f if line.strip())

def append_manifest(path, name):
    with open(path, "a") as f:
        f.write(name + "\n")

# ─── SESSION ──────────────────────────────────────────────────────────────────
class Session:
    def __init__(self, img, centers):
        self.img = img
        self.boxes = [{"cx": cx, "cy": cy, "state": GREY} for (cx, cy) in centers]
        self.scale = min(1.0,
                         MAX_DISPLAY_W / img.shape[1],
                         MAX_DISPLAY_H / (img.shape[0] + HEADER_H))
        self.mouse = (0, 0)   # last cursor pos in ORIGINAL image coords

    def to_orig(self, x, y):
        return int(x / self.scale), int(y / self.scale)

    def box_at(self, ox, oy, w, h):
        hw, hh = w // 2, h // 2
        best, bestd = None, 1e9
        for i, b in enumerate(self.boxes):
            if abs(ox - b["cx"]) <= hw and abs(oy - b["cy"]) <= hh:
                d = (ox - b["cx"])**2 + (oy - b["cy"])**2
                if d < bestd:
                    best, bestd = i, d
        return best

def render(sess, w, h, fname, idx, total, counts):
    vis = sess.img.copy()
    hw, hh = w // 2, h // 2
    for i, b in enumerate(sess.boxes):
        col = STATE_COLOR[b["state"]]
        cv2.rectangle(vis, (b["cx"]-hw, b["cy"]-hh), (b["cx"]+hw, b["cy"]+hh), col, 3)
        cv2.putText(vis, str(i+1), (b["cx"]-8, b["cy"]+6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    disp = cv2.resize(vis, None, fx=sess.scale, fy=sess.scale)
    g = sum(1 for b in sess.boxes if b["state"] == GOOD)
    d = sum(1 for b in sess.boxes if b["state"] == DEFECT)
    bar = np.zeros((HEADER_H, disp.shape[1], 3), np.uint8)
    cv2.putText(bar, f"[{idx+1}/{total}] {fname}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
    cv2.putText(bar, f"this img: good {g}  defect {d}  | crop {w}x{h}px | "
                     f"saved good {counts['good']} defect {counts['defect']}",
                (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    cv2.putText(bar, "L-click cycle/add  d delete  n next  p prev  +/- width  w/s height  r reset  q quit",
                (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,180), 1)
    return np.vstack([bar, disp])

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    global CROP_W, CROP_H

    for d in (OUTPUT_GOOD, OUTPUT_DEF):
        os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    if not os.path.isdir(DATASET_DIR):
        print(f"ERROR: dataset folder not found: {DATASET_DIR}")
        sys.exit(1)

    done = load_manifest(MANIFEST)

    files = []
    for sub in SRC_SUBDIRS:
        d = os.path.join(DATASET_DIR, sub)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                files.append(os.path.join(sub, f))

    todo = [f for f in files if f not in done]
    if not todo:
        print("All images already labeled (per manifest). Nothing to do.")
        print(f"Delete {MANIFEST} to relabel from scratch.")
        return

    print(f"{len(files)} images, {len(done)} done, {len(todo)} to label.")
    counts = {"good": 0, "defect": 0}
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

    idx = 0
    while 0 <= idx < len(todo):
        fname = todo[idx]
        img = cv2.imread(os.path.join(DATASET_DIR, fname))
        if img is None:
            print(f"skip unreadable {fname}"); idx += 1; continue
        stem = os.path.splitext(fname)[0].replace(os.sep, "_").replace("/", "_")
        sess = Session(img, detect_veins(img))

        def on_mouse(event, x, y, flags, param):
            oy = y - HEADER_H
            if oy < 0:
                return
            ox, oym = sess.to_orig(x, oy)
            sess.mouse = (ox, oym)
            if event == cv2.EVENT_LBUTTONDOWN:
                i = sess.box_at(ox, oym, CROP_W, CROP_H)
                if i is not None:
                    sess.boxes[i]["state"] = (sess.boxes[i]["state"] + 1) % 3
                else:
                    sess.boxes.append({"cx": ox, "cy": oym, "state": GREY})
            elif event == cv2.EVENT_RBUTTONDOWN:
                i = sess.box_at(ox, oym, CROP_W, CROP_H)
                if i is not None:
                    sess.boxes.pop(i)
        cv2.setMouseCallback(WIN, on_mouse)

        while True:
            cv2.imshow(WIN, render(sess, CROP_W, CROP_H, fname, idx, len(todo), counts))
            key = cv2.waitKey(20) & 0xFF

            if key in (ord('n'), ord(' '), 83, 13):
                s = save_crops(img, sess.boxes, stem, CROP_W, CROP_H)
                counts["good"] += s["good"]; counts["defect"] += s["defect"]
                if fname not in done:
                    append_manifest(MANIFEST, fname); done.add(fname)
                idx += 1
                break
            elif key in (ord('p'), 81):
                idx = max(0, idx - 1)
                break
            elif key in (ord('+'), ord('=')):
                CROP_W = min(CROP_MAX, CROP_W + CROP_STEP)
            elif key in (ord('-'), ord('_')):
                CROP_W = max(CROP_MIN, CROP_W - CROP_STEP)
            elif key == ord('w'):
                CROP_H = min(CROP_MAX, CROP_H + CROP_STEP)
            elif key == ord('s'):
                CROP_H = max(CROP_MIN, CROP_H - CROP_STEP)
            elif key == ord('d'):
                i = sess.box_at(sess.mouse[0], sess.mouse[1], CROP_W, CROP_H)
                if i is not None:
                    sess.boxes.pop(i)
            elif key == ord('r'):
                sess = Session(img, detect_veins(img))
                cv2.setMouseCallback(WIN, on_mouse)
            elif key in (ord('q'), 27):
                s = save_crops(img, sess.boxes, stem, CROP_W, CROP_H)
                counts["good"] += s["good"]; counts["defect"] += s["defect"]
                if fname not in done:
                    append_manifest(MANIFEST, fname)
                cv2.destroyAllWindows()
                print(f"\nStopped. This session — good {counts['good']}, defect {counts['defect']}.")
                return

    cv2.destroyAllWindows()
    print(f"\nDone. Session totals — good {counts['good']}, defect {counts['defect']}.")
    print(f"Crops in:\n  {OUTPUT_GOOD}\n  {OUTPUT_DEF}")

if __name__ == "__main__":
    main()