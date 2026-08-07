#!/usr/bin/env python3
"""
VisionGuard — Inspection Orchestrator v7
========================================
Live end-to-end inspection:
  STM32 motion + lighting → RPi5 capture → TCP image stream → AI inference
  → per-vein verdict aggregation → JSON report.

Single-wall (LEFT) PoC scope. 40 veins, 1 sweet-spot ROI per frame.
Drift correction: alternating MOV:124 / MOV:120 based on cumulative steps.
"""

import os
import sys
import tty
import termios
import serial
import socket
import time
import json
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ─── AI imports (visionguard/ai/src on path) ─────────────────────────────────
sys.path.insert(0, os.path.expanduser("~/visionguard/ai/src"))
from inference import Inferer, annotate, VERDICT_DEFECT, VERDICT_UNCERTAIN, VERDICT_NO_VEIN

# ─── Configuration ────────────────────────────────────────────────────────────
STM32_PORT      = "/dev/ttyACM0"
STM32_BAUD      = 115200
RPI5_HOST       = "192.168.10.2"
RPI5_PORT       = 9999
IMAGES_LOCAL    = os.path.expanduser("~/visionguard/inspection")
ROI_CONFIG      = os.path.expanduser("~/visionguard/ai/src/roi_config.json")
MODEL_PATH      = os.path.expanduser("~/visionguard/ai/models/resnet18_v2.pth")

TOTAL_VEINS     = 40
STEP_NORMAL     = 120
STEP_CORRECTED  = 124      # drift correction: every 2nd step
STEP_FINE       = 27
STABILIZE_DELAY = 0.3

LIGHT_MAX_DUTY  = 999
LIGHT_STEP      = LIGHT_MAX_DUTY // 9
def level_to_duty(level):
    return min(level * LIGHT_STEP, LIGHT_MAX_DUTY)

# ─── Terminal ─────────────────────────────────────────────────────────────────
def get_keypress():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == '\x1b':
            ch2 = sys.stdin.read(1)
            ch3 = sys.stdin.read(1)
            return ch + ch2 + ch3
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

# ─── STM32 ────────────────────────────────────────────────────────────────────
def stm32_connect():
    print("[STM32] Connecting...", end="", flush=True)
    ser = serial.Serial(STM32_PORT, STM32_BAUD, timeout=5)
    ser.setDTR(False); ser.setRTS(False)
    time.sleep(2)
    ser.reset_input_buffer()
    start = time.time()
    while time.time() - start < 5:
        if ser.in_waiting:
            if ser.readline().decode().strip() == "READY":
                print(" OK"); return ser
    print(" OK (no READY)")
    return ser

def stm32_send(ser, cmd, timeout=15):
    ser.write((cmd + "\n").encode())
    ser.timeout = timeout
    r = ser.readline().decode().strip()
    ser.timeout = 5
    return r

# ─── RPi5 TCP ─────────────────────────────────────────────────────────────────
# ─── RPi5 TCP ─────────────────────────────────────────────────────────────────
_RECV_BUFS = {}  # socket id -> leftover bytes

def rpi5_connect():
    print("[RPi5] Connecting...", end="", flush=True)
    sock = socket.socket()
    sock.connect((RPI5_HOST, RPI5_PORT))
    sock.settimeout(20)
    _RECV_BUFS[id(sock)] = b""
    print(" OK")
    return sock

def _recv_line(sock):
    """Read one '\\n'-terminated line from sock."""
    sid = id(sock)
    buf = _RECV_BUFS.get(sid, b"")
    while b"\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            break
        buf += chunk
    line, _, rest = buf.partition(b"\n")
    _RECV_BUFS[sid] = rest
    return line.decode("utf-8").strip()

def _recv_exact(sock, n):
    """Read exactly n bytes, consuming leftover buffer first."""
    sid = id(sock)
    buf = _RECV_BUFS.get(sid, b"")
    while len(buf) < n:
        chunk = sock.recv(min(65536, n - len(buf)))
        if not chunk:
            raise ConnectionError("RPi5 closed mid-stream")
        buf += chunk
    out, _RECV_BUFS[sid] = buf[:n], buf[n:]
    return out
def rpi5_capture_and_recv(sock, label, wall):
    """
    Send CAPTURE_SEND, receive header 'OK:filename:size\\n' + raw JPEG bytes.
    Returns (filename, np.ndarray BGR image) or (None, None) on error.
    """
    sock.sendall(f"CAPTURE_SEND:{label}:{wall}\n".encode())
    header = _recv_line(sock)
    if not header.startswith("OK:"):
        print(f"\n  [RPi5] {header}")
        return None, None
    try:
        _, filename, size = header.split(":")
        size = int(size)
    except ValueError:
        print(f"\n  [RPi5] bad header: {header}")
        return None, None
    img_bytes = _recv_exact(sock, size)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        print(f"\n  [RPi5] could not decode {filename}")
        return None, None
    return filename, img

def rpi5_send_simple(sock, cmd):
    """For QUIT etc. — text-only response."""
    sock.sendall((cmd + "\n").encode())
    return _recv_line(sock)

# ─── Load ROI ─────────────────────────────────────────────────────────────────
def load_roi():
    with open(ROI_CONFIG) as f:
        rc = json.load(f)
    # roi_config.json stores corners; vision.detect_vein_in_roi wants (x,y,w,h)
    x, y = rc["x1"], rc["y1"]
    w, h = rc["x2"] - rc["x1"], rc["y2"] - rc["y1"]
    return [(x, y, w, h)]   # list of one ROI

# ─── Manual Positioning ───────────────────────────────────────────────────────
def manual_positioning(ser):
    print("\n" + "═" * 50)
    print("  MANUAL POSITIONING")
    print("  ← →  Fine movement (~1°)")
    print("  ENTER  Confirm position")
    print("═" * 50)
    while True:
        print("\r  Use ← → to position. ENTER to confirm.    ", end="", flush=True)
        key = get_keypress()
        if key == '\x1b[C':
            r = stm32_send(ser, f"MOV:-{STEP_FINE}")
            print(f"\r  → +1°  [{r}]                            ", end="", flush=True)
        elif key == '\x1b[D':
            r = stm32_send(ser, f"MOV:{STEP_FINE}")
            print(f"\r  ← -1°  [{r}]                            ", end="", flush=True)
        elif key in ('\r', '\n'):
            print("\n  Position confirmed.")
            break
        elif key == '\x03':
            print("\n  Cancelled."); sys.exit(0)

# ─── Light setup ──────────────────────────────────────────────────────────────
def setup_lighting(ser):
    print("\n" + "═" * 50)
    print("  LIGHTING ADJUSTMENT")
    print("  0-9  Set light level")
    print("  ENTER  Confirm level")
    print("═" * 50)
    level = 6
    duty = level_to_duty(level)
    stm32_send(ser, f"LIGHT:SET:{duty}")
    print(f"  Initial level: {level}/9 (duty {duty})")
    while True:
        print(f"\r  Current level: {level}/9 (duty {duty})    Press 0-9 or ENTER    ", end="", flush=True)
        key = get_keypress()
        if key in '0123456789':
            level = int(key)
            duty = level_to_duty(level)
            r = stm32_send(ser, f"LIGHT:SET:{duty}")
            print(f"\r  Level {level}/9 (duty {duty})  [{r}]                      ", end="", flush=True)
        elif key in ('\r', '\n'):
            print(f"\n  Light locked at level {level}/9.")
            return level
        elif key == '\x03':
            stm32_send(ser, "LIGHT:OFF"); sys.exit(0)

# ─── Progress Bar ─────────────────────────────────────────────────────────────
def progress_bar(current, total, info="", width=20):
    filled  = int(width * current / total)
    bar     = "█" * filled + "░" * (width - filled)
    print(f"\r  [{bar}] {current:02d}/{total}  {info:<28}", end="", flush=True)

# ─── Inspection Cycle ─────────────────────────────────────────────────────────
def run_inspection(ser, sock, inferer, rois, light_level):
    date_str    = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = os.path.join(IMAGES_LOCAL, f"session_{date_str}")
    frames_dir  = os.path.join(session_dir, "frames")
    annot_dir   = os.path.join(session_dir, "annotated")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(annot_dir,  exist_ok=True)

    results    = []
    start_time = time.time()

    print("\n" + "═" * 50)
    print("  VISIONGUARD — INSPECTION CYCLE")
    print(f"  Session     : {date_str}")
    print(f"  Veins       : {TOTAL_VEINS}")
    print(f"  Light level : {light_level}/9")
    print(f"  Model       : {os.path.basename(MODEL_PATH)}")
    print(f"  ROI         : {rois[0]}")
    print("═" * 50 + "\n")

    for vein in range(1, TOTAL_VEINS + 1):
        # ── Move to next vein (skip move on first vein) ──────────────
        if vein > 1:
            # drift correction: every 2nd step gets the +4 microstep
            step = STEP_CORRECTED if (vein - 1) % 2 == 0 else STEP_NORMAL
            r = stm32_send(ser, f"MOV:{step}", timeout=15)
            if r != "OK":
                progress_bar(vein, TOTAL_VEINS, f"MOV ERR v{vein}")
                results.append({"vein": vein, "verdict": "mov_error", "p_defect": None})
                continue

        time.sleep(STABILIZE_DELAY)

        # ── Capture + receive image via TCP ──────────────────────────
        filename, img = rpi5_capture_and_recv(sock, "inspect", "L")
        if img is None:
            progress_bar(vein, TOTAL_VEINS, f"CAPTURE FAILED v{vein}")
            results.append({"vein": vein, "verdict": "capture_error", "p_defect": None})
            continue

        # save raw frame locally
        raw_path = os.path.join(frames_dir, f"v{vein:02d}_{filename}")
        cv2.imwrite(raw_path, img)

        # ── AI inference ─────────────────────────────────────────────
        res = inferer.classify_rois(img, rois)
        r0 = res[0]
        verdict = r0["verdict"]
        p_def   = r0["p_defect"]

        # save annotated frame
        annot = annotate(img, res, rois=rois)
        annot_path = os.path.join(annot_dir, f"v{vein:02d}_{verdict}.jpg")
        cv2.imwrite(annot_path, annot)

        results.append({
            "vein":       vein,
            "file":       os.path.basename(raw_path),
            "annotated":  os.path.basename(annot_path),
            "verdict":    verdict,
            "p_defect":   p_def,
            "cx":         r0["cx"],
            "cy":         r0["cy"],
        })

        p_str = f"{p_def:.2f}" if p_def is not None else "—"
        progress_bar(vein, TOTAL_VEINS, f"v{vein:02d} {verdict} p={p_str}")

    print()

    # ── Return to home position (one more MOV to bring vein 1 back) ──
    print("[HOME] Returning to vein 1...")
    step = STEP_CORRECTED if (TOTAL_VEINS) % 2 == 0 else STEP_NORMAL
    r = stm32_send(ser, f"MOV:{step}", timeout=15)
    print(f"  [{r}]")

    # ── Light OFF ────────────────────────────────────────────────────
    stm32_send(ser, "LIGHT:OFF")

    duration = time.time() - start_time

    # ── Report ───────────────────────────────────────────────────────
    defects   = [r for r in results if r["verdict"] == VERDICT_DEFECT]
    uncertain = [r for r in results if r["verdict"] == VERDICT_UNCERTAIN]
    no_vein   = [r for r in results if r["verdict"] == VERDICT_NO_VEIN]
    errors    = [r for r in results if r["verdict"] in ("mov_error", "capture_error")]
    good      = [r for r in results if r["verdict"] == "good"]

    report = {
        "timestamp":     date_str,
        "duration_s":    round(duration, 1),
        "total_veins":   TOTAL_VEINS,
        "model":         os.path.basename(MODEL_PATH),
        "light_level":   light_level,
        "roi":           rois[0],
        "counts": {
            "good":      len(good),
            "defect":    len(defects),
            "uncertain": len(uncertain),
            "no_vein":   len(no_vein),
            "errors":    len(errors),
        },
        "pass_fail":     "FAIL" if defects else ("REVIEW" if uncertain or no_vein else "PASS"),
        "defect_veins":  [r["vein"] for r in defects],
        "uncertain_veins": [r["vein"] for r in uncertain],
        "no_vein_veins": [r["vein"] for r in no_vein],
        "details":       results,
    }
    report_path = os.path.join(session_dir, f"report_{date_str}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # ── Summary print ────────────────────────────────────────────────
    print("\n" + "═" * 50)
    print(f"  INSPECTION COMPLETE — {report['pass_fail']}")
    print("─" * 50)
    print(f"  Duration   : {duration:.1f}s")
    print(f"  Good       : {len(good)}")
    print(f"  Defect     : {len(defects)}  {[r['vein'] for r in defects] if defects else ''}")
    print(f"  Uncertain  : {len(uncertain)}  {[r['vein'] for r in uncertain] if uncertain else ''}")
    print(f"  No vein    : {len(no_vein)}  {[r['vein'] for r in no_vein] if no_vein else ''}")
    print(f"  Errors     : {len(errors)}  {[r['vein'] for r in errors] if errors else ''}")
    print(f"  Report     : {report_path}")
    print(f"  Frames     : {frames_dir}")
    print(f"  Annotated  : {annot_dir}")
    print("═" * 50)

    subprocess.run(["paplay", "/usr/share/sounds/freedesktop/stereo/complete.oga"],
                   capture_output=True)
    return report

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(IMAGES_LOCAL, exist_ok=True)

    if not os.path.isfile(ROI_CONFIG):
        print(f"[FATAL] ROI config not found: {ROI_CONFIG}")
        print("        Run calibrate_roi or pick.py + roi_config.json first.")
        sys.exit(1)
    if not os.path.isfile(MODEL_PATH):
        print(f"[FATAL] Model not found: {MODEL_PATH}")
        sys.exit(1)

    print("\n" + "═" * 50)
    print("   VISIONGUARD — LIVE INSPECTION")
    print("═" * 50)

    rois = load_roi()
    print(f"[ROI ] Loaded: {rois[0]}")

    print("[AI  ] Loading model...", end="", flush=True)
    inferer = Inferer(model_path=MODEL_PATH)
    print(f" OK ({inferer.device})")

    ser  = stm32_connect()
    sock = rpi5_connect()

    try:
        light_level = setup_lighting(ser)
        manual_positioning(ser)
        

        print("\n" + "═" * 50)
        print("  READY — Press ENTER to start the 40-vein inspection cycle.")
        print("═" * 50)
        input()

        run_inspection(ser, sock, inferer, rois, light_level)

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        try: stm32_send(ser, "LIGHT:OFF")
        except: pass
        try: rpi5_send_simple(sock, "QUIT")
        except: pass
        ser.close()
        sock.close()
        print("  Connections closed.")

if __name__ == "__main__":
    main()