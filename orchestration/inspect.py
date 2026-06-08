#!/usr/bin/env python3
"""
VisionGuard — Inspection Orchestrator v3
Laptop orchestrates STM32 (motion+light) + RPi5 (capture) + AI inference stub.
"""

import os
import sys
import serial
import socket
import time
import json
import random
import subprocess
from datetime import datetime

# ─── Configuration ────────────────────────────────────────────────────────────
STM32_PORT      = "/dev/ttyACM0"
STM32_BAUD      = 115200
RPI5_HOST       = "192.168.10.2"
RPI5_PORT       = 9999
IMAGES_LOCAL    = os.path.expanduser("~/visionguard/inspection")
HOME_FILE       = os.path.expanduser("~/visionguard/tools/home.json")
TOTAL_VEINS     = 40
STEP_PER_VEIN   = 120
STABILIZE_DELAY = 0.3

# ─── STM32 ────────────────────────────────────────────────────────────────────
def stm32_connect():
    print("[STM32] Connecting...", end="", flush=True)
    ser = serial.Serial(STM32_PORT, STM32_BAUD, timeout=5)
    ser.setDTR(False)
    ser.setRTS(False)
    time.sleep(2)
    ser.reset_input_buffer()
    start = time.time()
    while time.time() - start < 5:
        if ser.in_waiting:
            line = ser.readline().decode().strip()
            if line == "READY":
                print(" OK")
                return ser
    print(" OK (no READY)")
    return ser

def stm32_send(ser, cmd, timeout=15):
    ser.write((cmd + "\n").encode())
    ser.timeout = timeout
    response = ser.readline().decode().strip()
    ser.timeout = 5
    return response

# ─── RPi5 ─────────────────────────────────────────────────────────────────────
def rpi5_connect():
    print("[RPi5] Connecting...", end="", flush=True)
    sock = socket.socket()
    sock.connect((RPI5_HOST, RPI5_PORT))
    sock.settimeout(15)
    print(" OK")
    return sock

def rpi5_send(sock, cmd):
    sock.sendall((cmd + "\n").encode())
    return sock.recv(1024).decode().strip()

# ─── AI Inference STUB ────────────────────────────────────────────────────────
def ai_predict(image_path: str) -> dict:
    """STUB — replace with real model when available."""
    result = random.choice(["OK", "OK", "OK", "NOK"])
    return {
        "result": result,
        "confidence": round(random.uniform(0.75, 0.99), 2),
        "model": "stub"
    }

# ─── Home ─────────────────────────────────────────────────────────────────────
def load_home():
    if os.path.exists(HOME_FILE):
        with open(HOME_FILE) as f:
            return json.load(f)
    return None

# ─── SCP Transfer ─────────────────────────────────────────────────────────────
def transfer_images(session_dir: str):
    print("\n[SCP] Transferring images from RPi5...")
    os.makedirs(session_dir, exist_ok=True)
    cmd = [
        "scp", "-r",
        f"pi5@{RPI5_HOST}:/home/pi5/visionguard/inspection/",
        session_dir
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("[SCP] Transfer complete.")
        subprocess.run([
            "ssh", f"pi5@{RPI5_HOST}",
            "rm -rf /home/pi5/visionguard/inspection/*"
        ])
    else:
        print(f"[SCP] Failed: {result.stderr}")

# ─── Report ───────────────────────────────────────────────────────────────────
def generate_report(results: list, session_dir: str, duration: float):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": ts,
        "duration_seconds": round(duration, 1),
        "total_veins": TOTAL_VEINS,
        "total_images": len(results),
        "ok": sum(1 for r in results if r["result"] == "OK"),
        "nok": sum(1 for r in results if r["result"] == "NOK"),
        "model": "stub",
        "details": results
    }
    report_path = os.path.join(session_dir, f"report_{ts}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Report] Saved: {report_path}")
    return report

# ─── Display ──────────────────────────────────────────────────────────────────
def print_status(phase, vein, wall, speed, captured, last_file, last_result):
    print("\n" + "─" * 50)
    print(f"  Phase    : {phase}")
    print(f"  Vein     : {vein:02d} / {TOTAL_VEINS}")
    print(f"  Wall     : {wall}")
    print(f"  Speed    : {speed} steps/sec")
    print(f"  Captured : {captured}")
    print(f"  Last     : {last_file}")
    print(f"  Result   : {last_result}")
    print("─" * 50)

# ─── Inspection Cycle ─────────────────────────────────────────────────────────
def run_inspection(ser, sock):
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(IMAGES_LOCAL, f"session_{ts}")
    os.makedirs(session_dir, exist_ok=True)
    results     = []
    speed       = 800
    start_time  = time.time()

    print("\n" + "═" * 50)
    print("  VISIONGUARD — INSPECTION CYCLE")
    print(f"  Session  : {ts}")
    print(f"  Veins    : {TOTAL_VEINS}")
    print(f"  Speed    : {speed} steps/sec")
    print(f"  Step/vein: {STEP_PER_VEIN}")
    print("═" * 50)

    # ── LIGHT ON ──────────────────────────────────────────────
    print("\n[LIGHT] Turning ON...")
    stm32_send(ser, "LIGHT:ON")
    time.sleep(0.2)

    # ══════════════════════════════════════════════════════════
    # TOUR 1 — LEFT WALLS
    # ══════════════════════════════════════════════════════════
    print("\n[Tour 1] LEFT walls — starting...")
    captured_left = 0

    for vein in range(1, TOTAL_VEINS + 1):

        # Move (skip on vein 1 — already at home)
        if vein > 1:
            r = stm32_send(ser, f"MOV:{STEP_PER_VEIN}", timeout=15)
            if r != "OK":
                print(f"  [ERR] MOV failed vein {vein}: {r}")
                continue

        time.sleep(STABILIZE_DELAY)

        # Capture
        r = rpi5_send(sock, "CAPTURE:inspect:L")
        if r.startswith("OK:"):
            filename = r.split(":", 1)[1]
            captured_left += 1
            prediction = ai_predict(filename)
            results.append({
                "vein": vein,
                "wall": "LEFT",
                "file": filename,
                **prediction
            })
            print_status(
                phase="Tour 1 — LEFT",
                vein=vein,
                wall="LEFT",
                speed=speed,
                captured=captured_left,
                last_file=filename,
                last_result=f"{prediction['result']} ({prediction['confidence']*100:.1f}%)"
            )
        else:
            print(f"  [ERR] Capture failed vein {vein} LEFT: {r}")

    # ── RETURN TO HOME ────────────────────────────────────────
    steps_back = -((TOTAL_VEINS - 1) * STEP_PER_VEIN)
    print(f"\n[Return] Going back to home ({abs(steps_back)} steps)...")
    r = stm32_send(ser, f"MOV:{steps_back}", timeout=30)
    print(f"  [{r}]")
    time.sleep(0.5)

    # ── CAMERA SWITCH CONFIRMATION ────────────────────────────
    print("\n" + "═" * 50)
    print("  CAMERA REPOSITIONING REQUIRED")
    print("  → Move camera to RIGHT wall position")
    print(f"  LEFT images captured : {captured_left}")
    print(f"  Speed                : {speed} steps/sec")
    print(f"  Elapsed              : {round(time.time()-start_time, 1)}s")
    print("═" * 50)
    input("\n  Press ENTER when camera is in RIGHT position...")

    # ══════════════════════════════════════════════════════════
    # TOUR 2 — RIGHT WALLS
    # ══════════════════════════════════════════════════════════
    print("\n[Tour 2] RIGHT walls — starting...")
    captured_right = 0

    for vein in range(1, TOTAL_VEINS + 1):

        if vein > 1:
            r = stm32_send(ser, f"MOV:{STEP_PER_VEIN}", timeout=15)
            if r != "OK":
                print(f"  [ERR] MOV failed vein {vein}: {r}")
                continue

        time.sleep(STABILIZE_DELAY)

        r = rpi5_send(sock, "CAPTURE:inspect:R")
        if r.startswith("OK:"):
            filename = r.split(":", 1)[1]
            captured_right += 1
            prediction = ai_predict(filename)
            results.append({
                "vein": vein,
                "wall": "RIGHT",
                "file": filename,
                **prediction
            })
            print_status(
                phase="Tour 2 — RIGHT",
                vein=vein,
                wall="RIGHT",
                speed=speed,
                captured=captured_right,
                last_file=filename,
                last_result=f"{prediction['result']} ({prediction['confidence']*100:.1f}%)"
            )
        else:
            print(f"  [ERR] Capture failed vein {vein} RIGHT: {r}")

    # ── RETURN TO HOME ────────────────────────────────────────
    steps_back = -((TOTAL_VEINS - 1) * STEP_PER_VEIN)
    print(f"\n[Return] Going back to home ({abs(steps_back)} steps)...")
    r = stm32_send(ser, f"MOV:{steps_back}", timeout=30)
    print(f"  [{r}]")
    time.sleep(0.5)

    # ── LIGHT OFF ─────────────────────────────────────────────
    print("\n[LIGHT] Turning OFF...")
    stm32_send(ser, "LIGHT:OFF")

    # ── TRANSFER + REPORT ─────────────────────────────────────
    duration = time.time() - start_time
    transfer_images(session_dir)
    report = generate_report(results, session_dir, duration)

    # ── FINAL SUMMARY ─────────────────────────────────────────
    print("\n" + "═" * 50)
    print(f"  INSPECTION COMPLETE")
    print(f"  Duration : {round(duration, 1)}s")
    print(f"  Images   : {report['total_images']} / {TOTAL_VEINS * 2}")
    print(f"  OK       : {report['ok']}")
    print(f"  NOK      : {report['nok']}")
    print(f"  Session  : {session_dir}")
    print("═" * 50)

    return report

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(IMAGES_LOCAL, exist_ok=True)

    # ── Check home ────────────────────────────────────────────
    home = load_home()
    if not home:
        print("\n[ERROR] No home.json found.")
        print("  Run set_home.py first to calibrate home position.")
        sys.exit(1)

    print(f"\n[HOME] Home position loaded — set on {home['timestamp']}")

    ser  = stm32_connect()
    sock = rpi5_connect()

    print("\n" + "═" * 50)
    print("  SYSTEM READY")
    print("  Make sure:")
    print("  → Motor is at HOME position")
    print("  → Camera faces LEFT wall")
    print("  → Lighting circuit connected")
    print("═" * 50)

    try:
        input("\n  Press ENTER to start inspection cycle...")
        run_inspection(ser, sock)
    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        stm32_send(ser, "LIGHT:OFF")
        ser.close()
        sock.close()
        print("  Connections closed.")

if __name__ == "__main__":
    main()