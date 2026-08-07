"""
VisionGuard — Shared Vision Utilities
=====================================
Vein detection + cropping, shared by the labeling tool AND live inference so
both use IDENTICAL logic. Detection parameters live in config.py.

Functions:
  detect_veins(img)            -> list of (cx, cy) centers across whole frame.
                                   Used in offline DETECT-mode testing.

  detect_vein_in_roi(img, roi) -> single (cx, cy) of the strongest vein inside
                                   an ROI box, or None. Used by inference to
                                   precisely center on a vein rotated into a
                                   known sweet-spot region (absorbs mechanical
                                   drift). ROI is the search zone; the actual
                                   crop is always fixed at (CROP_W, CROP_H).

  crop_at(img, cx, cy, w, h)   -> fixed-size crop centered on (cx, cy),
                                   zero-padded at borders. Must produce the
                                   exact same crop geometry the model was
                                   trained on.
"""

import cv2
import numpy as np

import config as C


# ─── internal: vein mask ──────────────────────────────────────────────────────
def _vein_mask(gray):
    """Binary mask isolating bright lit-vein blobs from the matte background.

    Threshold + morphological open/close cleans up small specks and bridges
    micro-gaps inside one vein. Tuned via DET_* params in config.
    """
    _, mask = cv2.threshold(gray, C.DET_THRESH, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (C.DET_MORPH_K, C.DET_MORPH_K))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


# ─── whole-frame detection (offline DETECT mode) ──────────────────────────────
def detect_veins(img):
    """Return [(cx, cy), ...] of all valid vein blobs in the frame,
    sorted left-to-right.

    Filters:
      - area within [DET_AREA_MIN, DET_AREA_MAX]   (rejects specks + giant blobs)
      - aspect ratio >= DET_AR_MIN                 (rejects round shapes, e.g. screws)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = _vein_mask(gray)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < C.DET_AREA_MIN or a > C.DET_AREA_MAX:
            continue
        x, y, w, h = cv2.boundingRect(c)
        ar = max(w, h) / max(1, min(w, h))
        if ar < C.DET_AR_MIN:
            continue
        centers.append((x + w // 2, y + h // 2))

    centers.sort(key=lambda p: p[0])
    return centers


# ─── ROI-local detection (live inference centering) ──────────────────────────
def detect_vein_in_roi(img, roi):
    """Locate the single strongest vein inside an ROI search zone.

    Returns its center (cx, cy) in FULL-IMAGE coordinates, or None if no valid
    vein is found inside the ROI (drift too large, vein occluded, etc.).

    Args:
        img: full BGR frame.
        roi: (x, y, w, h) search zone in full-image coordinates. Should be
             larger than (CROP_W, CROP_H) to absorb mechanical drift while
             being narrower than the vein-to-vein spacing to avoid picking
             the wrong vein.

    Strategy:
        crop the ROI, run the same masking + filters as detect_veins, return
        the largest valid blob (or None). Coordinates are translated back to
        the full-image frame before returning, so downstream crop_at() works
        directly without any further offset math.
    """
    rx, ry, rw, rh = roi
    H, W = img.shape[:2]

    # clamp ROI to image bounds (defensive — calibration may overshoot edges)
    rx = max(0, rx); ry = max(0, ry)
    rw = min(rw, W - rx); rh = min(rh, H - ry)
    if rw <= 0 or rh <= 0:
        return None

    sub = img[ry:ry + rh, rx:rx + rw]
    if sub.size == 0:
        return None

    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    mask = _vein_mask(gray)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Score = area weighted by proximity to ROI center
    # — favors veins that are both bright AND well-centered in the ROI,
    # avoiding partial slivers near the ROI edges.
    roi_cx = rx + rw // 2
    roi_cy = ry + rh // 2
    best_center, best_score = None, -1.0
    for c in cnts:
        a = cv2.contourArea(c)
        if a < C.DET_AREA_MIN or a > C.DET_AREA_MAX:
            continue
        x, y, w, h = cv2.boundingRect(c)
        ar = max(w, h) / max(1, min(w, h))
        if ar < C.DET_AR_MIN:
            continue
        bcx = rx + x + w // 2
        bcy = ry + y + h // 2
        dist = ((bcx - roi_cx) ** 2 + (bcy - roi_cy) ** 2) ** 0.5
        score = a / (1.0 + dist)
        if score > best_score:
            best_score = score
            best_center = (bcx, bcy)

    return best_center


# ─── fixed-size crop (model input geometry) ───────────────────────────────────
def crop_at(img, cx, cy, w, h):
    """Return a fixed-size (w × h) crop centered on (cx, cy).

    Borders are zero-padded (black) when the crop would fall outside the
    image. The crop is the EXACT geometry the model was trained on — never
    resize this elsewhere; preprocessing inside dataset.preprocess_crop()
    handles the model-input resize.
    """
    H, W = img.shape[:2]
    hw, hh = w // 2, h // 2
    x1, y1 = max(0, cx - hw), max(0, cy - hh)
    x2, y2 = min(W, cx + hw), min(H, cy + hh)
    crop = img[y1:y2, x1:x2]

    pw, ph = w - (x2 - x1), h - (y2 - y1)
    if pw > 0 or ph > 0:
        crop = cv2.copyMakeBorder(crop, 0, ph, 0, pw,
                                  cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return crop