"""
VisionGuard — Inference
=======================
Loads the trained ResNet18 once and classifies vein crops from a frame.

Two operating modes (same code path, same result schema):

  • DETECT mode  — vision.detect_veins() across the whole frame, classify every
                   detected vein. For testing on existing Pursuit images.

  • ROI mode     — for each sweet-spot ROI passed in, run
                   vision.detect_vein_in_roi() to precisely locate the vein
                   (absorbing mechanical drift), then classify. This is what
                   the live orchestration uses (~13-20 rotations × 2-3 ROIs).

Three guarantees the script keeps:

  1. Preprocessing identical to training — imports preprocess_crop from
     dataset.py. Never reimplemented here.
  2. Class order from config — uses C.CLASSES.index("defect") instead of a
     hardcoded column, so if class order ever flips, inference still picks the
     correct probability column.
  3. Strict state_dict load — fails loudly on architecture mismatch instead of
     silently loading a broken model.

Verdict mapping (3-way classification + 1 detection failure case):
    P(defect) >= DEFECT_THRESH        -> "defect"      (red)
    P(defect) <= GOOD_THRESH          -> "good"        (green)
    in between                        -> "uncertain"   (yellow, human review)
    detect_vein_in_roi returned None  -> "no_vein"     (grey, ROI mode only)

CLI:
    python inference.py --image path/to/frame.jpg
    python inference.py --image f.jpg --rois "120,80,180,160;420,80,180,160"
    python inference.py --image f.jpg --json out.json --out f_annot.jpg
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

import config as C
from dataset import preprocess_crop
import vision


# ─── verdicts & colors ────────────────────────────────────────────────────────
VERDICT_GOOD      = "good"
VERDICT_DEFECT    = "defect"
VERDICT_UNCERTAIN = "uncertain"
VERDICT_NO_VEIN   = "no_vein"

VERDICT_COLOR = {                 # BGR
    VERDICT_GOOD:      (0, 200,   0),     # green
    VERDICT_DEFECT:    (0,   0, 230),     # red
    VERDICT_UNCERTAIN: (0, 220, 230),     # yellow
    VERDICT_NO_VEIN:   (180, 180, 180),   # grey
}

# Index of the "defect" probability column from config — never hardcode.
DEFECT_IDX = C.CLASSES.index("defect")


def verdict_from_prob(p_defect):
    """Apply the locked 3-way threshold policy."""
    if p_defect >= C.DEFECT_THRESH:
        return VERDICT_DEFECT
    if p_defect <= C.GOOD_THRESH:
        return VERDICT_GOOD
    return VERDICT_UNCERTAIN


# ─── inferer ──────────────────────────────────────────────────────────────────
class Inferer:
    """
    Holds the model + device. Instantiate ONCE per process (the live
    orchestrator does this at startup), then call classify_* per captured
    frame. Reloading per frame would add ~200-500 ms latency.

    Public:
        classify_frame(img_bgr)        -> list[dict]   (DETECT mode)
        classify_rois(img_bgr, rois)   -> list[dict]   (ROI / live mode)

    Result schema (one dict per vein / ROI):
        {
          "id":       int,            # index in the returned list
          "roi_idx":  int|None,       # ROI index (ROI mode) or None (DETECT)
          "cx":       int|None,       # vein center x in full-image coords
          "cy":       int|None,       # vein center y in full-image coords
          "p_defect": float|None,     # P(defect) ∈ [0,1], None if no_vein
          "verdict":  str,            # one of VERDICT_*
        }
    """

    def __init__(self, model_path=None, device=None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_path = model_path or C.INFER_MODEL_PATH
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"model not found: {self.model_path}")

        self.model = self._build_model()
        state = torch.load(self.model_path, map_location=self.device)
        # strict=True catches any architecture/key drift — surface it now,
        # not silently 50% accuracy later.
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    def _build_model(self):
        """Re-create the exact training architecture (no pretrained download
        needed — the state_dict overrides every weight)."""
        if C.MODEL_NAME == "resnet18":
            m = models.resnet18(weights=None)
            m.fc = nn.Linear(m.fc.in_features, C.NUM_CLASSES)
        elif C.MODEL_NAME == "mobilenet_v2":
            m = models.mobilenet_v2(weights=None)
            m.classifier[1] = nn.Linear(m.classifier[1].in_features,
                                        C.NUM_CLASSES)
        else:
            raise ValueError(f"unknown MODEL_NAME: {C.MODEL_NAME}")
        return m

    @torch.no_grad()
    def _classify_crops(self, crops_bgr):
        """
        Batch-classify a list of BGR crops.  Returns ndarray of P(defect),
        shape (N,), aligned with input order.

        Using ONE forward pass for all crops in a frame is meaningfully faster
        than per-crop on CPU (~3-5× for 8-10 crops).
        """
        if not crops_bgr:
            return np.empty((0,), dtype=np.float32)
        tensors = [preprocess_crop(c) for c in crops_bgr]
        batch = torch.stack(tensors, dim=0).to(self.device)
        logits = self.model(batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs[:, DEFECT_IDX].astype(np.float32)

    # ── DETECT mode ──────────────────────────────────────────────────────────
    def classify_frame(self, img_bgr):
        """Detect all veins in the frame and classify each. Used for static
        testing on existing Pursuit images, NOT for live (drift makes whole-
        frame detection unreliable — that's why ROI mode exists)."""
        centers = vision.detect_veins(img_bgr)
        crops = [vision.crop_at(img_bgr, cx, cy, C.CROP_W, C.CROP_H)
                 for (cx, cy) in centers]
        ps = self._classify_crops(crops)
        return [
            {
                "id": i,
                "roi_idx": None,
                "cx": int(cx),
                "cy": int(cy),
                "p_defect": float(p),
                "verdict": verdict_from_prob(float(p)),
            }
            for i, ((cx, cy), p) in enumerate(zip(centers, ps))
        ]

    # ── ROI / live mode ──────────────────────────────────────────────────────
    def classify_rois(self, img_bgr, rois):
        """
        For each (x, y, w, h) ROI, find the strongest vein inside and classify.

        ROIs where no vein is found return a "no_vein" verdict (not a crash).
        The live orchestrator needs to distinguish "vein found and is
        uncertain" from "vein not found at all" — they trigger different
        responses (flag for human vs. retry capture / re-home).
        """
        crops, meta = [], []
        for ri, roi in enumerate(rois):
            center = vision.detect_vein_in_roi(img_bgr, roi)
            if center is None:
                meta.append((ri, None))
                continue
            cx, cy = center
            crop = vision.crop_at(img_bgr, cx, cy, C.CROP_W, C.CROP_H)
            crops.append(crop)
            meta.append((ri, (cx, cy)))

        ps = self._classify_crops(crops)
        p_iter = iter(ps.tolist())

        results = []
        for ri, center in meta:
            if center is None:
                results.append({
                    "id": ri,
                    "roi_idx": ri,
                    "cx": None,
                    "cy": None,
                    "p_defect": None,
                    "verdict": VERDICT_NO_VEIN,
                })
            else:
                p = float(next(p_iter))
                results.append({
                    "id": ri,
                    "roi_idx": ri,
                    "cx": int(center[0]),
                    "cy": int(center[1]),
                    "p_defect": p,
                    "verdict": verdict_from_prob(p),
                })
        return results


# ─── annotation ───────────────────────────────────────────────────────────────
def annotate(img_bgr, results, rois=None, crop_w=None, crop_h=None):
    """Draw verdict-colored boxes + labels for each classified vein, plus
    grey ROI rectangles when ROI mode is used."""
    out = img_bgr.copy()
    H, W = out.shape[:2]
    cw = crop_w if crop_w is not None else C.CROP_W
    ch = crop_h if crop_h is not None else C.CROP_H

    # ROI rectangles first (behind everything)
    if rois:
        for ri, (x, y, w, h) in enumerate(rois):
            cv2.rectangle(out, (x, y), (x + w, y + h), (120, 120, 120), 1)
            cv2.putText(out, f"ROI{ri}", (x + 2, y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (120, 120, 120), 1, cv2.LINE_AA)

    for r in results:
        color = VERDICT_COLOR[r["verdict"]]

        if r["verdict"] == VERDICT_NO_VEIN:
            # mark the ROI center with an X — no crop to draw
            if rois is not None and r["roi_idx"] is not None:
                x, y, w, h = rois[r["roi_idx"]]
                cv2.line(out, (x, y), (x + w, y + h), color, 1)
                cv2.line(out, (x + w, y), (x, y + h), color, 1)
            continue

        cx, cy = r["cx"], r["cy"]
        x1 = max(0, cx - cw // 2); y1 = max(0, cy - ch // 2)
        x2 = min(W, cx + cw // 2); y2 = min(H, cy + ch // 2)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"{r['verdict']} {r['p_defect']:.2f}"
        cv2.putText(out, label, (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


# ─── CLI helpers ──────────────────────────────────────────────────────────────
def _parse_rois(s):
    """Parse '--rois "x,y,w,h;x,y,w,h;..."' into a list of int tuples."""
    if not s:
        return None
    rois = []
    for tok in s.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        parts = [int(v.strip()) for v in tok.split(",")]
        if len(parts) != 4:
            raise ValueError(f"bad ROI spec '{tok}': expected x,y,w,h")
        rois.append(tuple(parts))
    return rois


def _print_summary(results, rois, elapsed_ms, device, model_path, img_path):
    print(f"[inference] device={device}  model={model_path}")
    print(f"[inference] image={img_path}  "
          f"mode={'ROI' if rois else 'DETECT'}  "
          f"veins={len(results)}  elapsed={elapsed_ms:.1f} ms")
    counts = {VERDICT_GOOD: 0, VERDICT_DEFECT: 0,
              VERDICT_UNCERTAIN: 0, VERDICT_NO_VEIN: 0}
    for r in results:
        counts[r["verdict"]] += 1
        if r["verdict"] == VERDICT_NO_VEIN:
            print(f"  roi{r['roi_idx']}: NO VEIN FOUND")
        else:
            print(f"  v{r['id']:02d}  ({r['cx']:4d},{r['cy']:4d})  "
                  f"P(defect)={r['p_defect']:.3f}  -> {r['verdict']}")
    print(f"[inference] summary  good={counts[VERDICT_GOOD]}  "
          f"defect={counts[VERDICT_DEFECT]}  "
          f"uncertain={counts[VERDICT_UNCERTAIN]}  "
          f"no_vein={counts[VERDICT_NO_VEIN]}")


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="VisionGuard single-frame inference "
                    "(DETECT mode by default, ROI mode if --rois given)")
    ap.add_argument("--image", required=True, help="input frame path")
    ap.add_argument("--out", default=None,
                    help="annotated output image (default: <image>_annot.jpg)")
    ap.add_argument("--json", default=None,
                    help="JSON results path (optional)")
    ap.add_argument("--rois", default=None,
                    help='ROI list "x,y,w,h;x,y,w,h" — enables ROI mode')
    ap.add_argument("--model", default=None,
                    help="override INFER_MODEL_PATH")
    ap.add_argument("--device", default=None,
                    help="cuda|cpu (auto-detected if omitted)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"could not read {args.image}", file=sys.stderr)
        sys.exit(1)

    rois = _parse_rois(args.rois)

    inf = Inferer(model_path=args.model, device=args.device)

    t0 = time.time()
    if rois is None:
        results = inf.classify_frame(img)
    else:
        results = inf.classify_rois(img, rois)
    elapsed_ms = (time.time() - t0) * 1000.0

    annot = annotate(img, results, rois=rois)
    out_path = args.out or (str(Path(args.image).with_suffix("")) + "_annot.jpg")
    cv2.imwrite(out_path, annot)

    if args.json:
        payload = {
            "image": args.image,
            "model": str(inf.model_path),
            "device": str(inf.device),
            "mode": "ROI" if rois else "DETECT",
            "rois": rois,
            "thresholds": {"defect": C.DEFECT_THRESH, "good": C.GOOD_THRESH},
            "results": results,
            "elapsed_ms": elapsed_ms,
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)

    if not args.quiet:
        _print_summary(results, rois, elapsed_ms,
                       inf.device, inf.model_path, args.image)
        print(f"[inference] annotated -> {out_path}")
        if args.json:
            print(f"[inference] json      -> {args.json}")


if __name__ == "__main__":
    main()