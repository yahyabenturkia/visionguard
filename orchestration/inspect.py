#!/usr/bin/env python3
"""
VisionGuard — Inspection Orchestrator v6
"""

import os
import sys
import tty
import termios
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
TOTAL_VEINS     = 40
STEP_PER_VEIN   = 120
STEP_FINE       = 27
STABILIZE_DELAY = 0.3
RETRY_DELAY     = 1.0

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

def rpi5_capture(sock, cmd):
    """Capture with 1 automatic retry."""
    r = rpi5_send(sock, cmd)
    if r.startswith("OK:"):
        return r, False
    print(f"\n  [RETRY] Capture failed → retrying in {RETRY_DELAY}s...")
    time.sleep(RETRY_DELAY)
    r = rpi5_send(sock, cmd)
    if r.startswith("OK:"):
        return r, False
    return r, True  # True = failed after retry

# ─── AI Inference STUB ────────────────────────────────────────────────────────
def ai_predict(image_path: str) -> dict:
    result = random.choice(["OK", "OK", "OK", "NOK"])
    return {
        "result": result,
        "confidence": round(random.uniform(0.75, 0.99), 2),
        "model": "stub"
    }

# ─── Manual Positioning ───────────────────────────────────────────────────────
def manual_positioning(ser):
    print("\n" + "═" * 50)
    print("  MANUAL POSITIONING")
    print("  ← →  Fine movement (~1°)")
    print("  ENTER  Confirm home position")
    print("═" * 50)

    while True:
        print("\r  Use ← → to position. ENTER to confirm.", end="", flush=True)
        key = get_keypress()

        if key == '\x1b[C':
            r = stm32_send(ser, f"MOV:{STEP_FINE}")
            print(f"\r  → +1°  [{r}]                    ", end="", flush=True)

        elif key == '\x1b[D':
            r = stm32_send(ser, f"MOV:-{STEP_FINE}")
            print(f"\r  ← -1°  [{r}]                    ", end="", flush=True)

        elif key in ('\r', '\n'):
            print("\n  [HOME] Position confirmed.")
            break

        elif key == '\x03':
            print("\n  Cancelled.")
            sys.exit(0)

# ─── Progress Bar ─────────────────────────────────────────────────────────────
def progress_bar(current, total, result="", width=20):
    filled  = int(width * current / total)
    bar     = "█" * filled + "░" * (width - filled)
    percent = int(100 * current / total)
    print(f"\r  [{bar}] {current:02d}/{total}  {result:<20}", end="", flush=True)

# ─── SCP Transfer ─────────────────────────────────────────────────────────────
def transfer_and_clear(session_dir: str, wall: str):
    print(f"\n[SCP] Transferring {wall} wall images...")
    wall_dir = os.path.join(session_dir, wall.lower())
    os.makedirs(wall_dir, exist_ok=True)
    cmd = [
        "scp", "-r",
        f"pi5@{RPI5_HOST}:/home/pi5/inspection/",
        wall_dir
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[SCP] {wall} transfer complete.")
        subprocess.run([
            "ssh", f"pi5@{RPI5_HOST}",
            "rm -rf /home/pi5/inspection/*"
        ])
        print(f"[SCP] RPi5 cleared.")
    else:
        print(f"[SCP] Failed: {result.stderr}")

# ─── Report ───────────────────────────────────────────────────────────────────
def generate_report(results: list, session_dir: str, duration: float):
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    nok_list = [r for r in results if r["result"] == "NOK"]
    report   = {
        "timestamp"       : ts,
        "duration_seconds": round(duration, 1),
        "total_veins"     : TOTAL_VEINS,
        "total_images"    : len(results),
        "ok"              : sum(1 for r in results if r["result"] == "OK"),
        "nok"             : len(nok_list),
        "nok_veins"       : [
            {
                "vein"      : r["vein"],
                "wall"      : r["wall"],
                "file"      : r["file"],
                "confidence": r["confidence"]
            }
            for r in nok_list
        ],
        "model"  : "stub",
        "details": results
    }
    report_path = os.path.join(session_dir, f"report_{ts}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Report] Saved: {report_path}")
    return report

# ─── NOK Summary ──────────────────────────────────────────────────────────────
def print_nok_summary(nok_list):
    print("\n" + "═" * 50)
    print(f"  NOK SUMMARY — {len(nok_list)} defect(s) found")
    print("─" * 50)
    if not nok_list:
        print("  ✔ All veins passed inspection.")
    else:
        for item in nok_list:
            print(f"  ✘ Vein {item['vein']:02d} | {item['wall']:<5} | "
                  f"{item['file']} | "
                  f"confidence: {item['confidence']*100:.1f}%")
    print("═" * 50)

# ─── Inspection Cycle ─────────────────────────────────────────────────────────
def run_inspection(ser, sock):
    date_str    = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = os.path.join(IMAGES_LOCAL, f"session_{date_str}")
    os.makedirs(session_dir, exist_ok=True)
    results     = []
    speed       = 800
    start_time  = time.time()

    print("\n" + "═" * 50)
    print("  VISIONGUARD — INSPECTION CYCLE")
    print(f"  Session  : {date_str}")
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
    print("\n[Tour 1] LEFT walls\n")
    captured_left = 0

    for vein in range(1, TOTAL_VEINS + 1):

        if vein > 1:
            r = stm32_send(ser, f"MOV:{STEP_PER_VEIN}", timeout=15)
            if r != "OK":
                progress_bar(vein, TOTAL_VEINS, f"ERR MOV v{vein}")
                continue

        time.sleep(STABILIZE_DELAY)

        r, failed = rpi5_capture(sock, "CAPTURE:inspect:L")
        if not failed:
            filename = r.split(":", 1)[1]
            captured_left += 1
            prediction = ai_predict(filename)
            results.append({
                "vein": vein,
                "wall": "LEFT",
                "file": filename,
                **prediction
            })
            progress_bar(
                vein, TOTAL_VEINS,
                f"{prediction['result']} ({prediction['confidence']*100:.0f}%)"
            )
        else:
            progress_bar(vein, TOTAL_VEINS, f"FAILED v{vein}")

    print(f"\n  Done — {captured_left}/{TOTAL_VEINS} captured")

    # ── SCP Tour 1 ────────────────────────────────────────────
    transfer_and_clear(session_dir, "LEFT")

    # ── CAMERA SWITCH ─────────────────────────────────────────
    print("\n" + "═" * 50)
    print("  CAMERA REPOSITIONING REQUIRED")
    print("  → Move camera to RIGHT wall position")
    print(f"  LEFT captured : {captured_left} / {TOTAL_VEINS}")
    print(f"  Elapsed       : {round(time.time()-start_time, 1)}s")
    print("═" * 50)
    input("\n  Press ENTER when camera is in RIGHT position...")

    # ══════════════════════════════════════════════════════════
    # TOUR 2 — RIGHT WALLS
    # ══════════════════════════════════════════════════════════
    print("\n[Tour 2] RIGHT walls\n")
    captured_right = 0

    for vein in range(1, TOTAL_VEINS + 1):

        if vein > 1:
            r = stm32_send(ser, f"MOV:{STEP_PER_VEIN}", timeout=15)
            if r != "OK":
                progress_bar(vein, TOTAL_VEINS, f"ERR MOV v{vein}")
                continue

        time.sleep(STABILIZE_DELAY)

        r, failed = rpi5_capture(sock, "CAPTURE:inspect:R")
        if not failed:
            filename = r.split(":", 1)[1]
            captured_right += 1
            prediction = ai_predict(filename)
            results.append({
                "vein": vein,
                "wall": "RIGHT",
                "file": filename,
                **prediction
            })
            progress_bar(
                vein, TOTAL_VEINS,
                f"{prediction['result']} ({prediction['confidence']*100:.0f}%)"
            )
        else:
            progress_bar(vein, TOTAL_VEINS, f"FAILED v{vein}")

    print(f"\n  Done — {captured_right}/{TOTAL_VEINS} captured")

    # ── RETURN TO HOME (MOV:120 after vein 40) ────────────────
    print("\n[HOME] Returning to vein 1...")
    r = stm32_send(ser, f"MOV:{STEP_PER_VEIN}", timeout=15)
    print(f"  [{r}]")

    # ── SCP Tour 2 ────────────────────────────────────────────
    transfer_and_clear(session_dir, "RIGHT")

    # ── LIGHT OFF ─────────────────────────────────────────────
    print("\n[LIGHT] Turning OFF...")
    stm32_send(ser, "LIGHT:OFF")

    # ── REPORT ────────────────────────────────────────────────
    duration = time.time() - start_time
    report   = generate_report(results, session_dir, duration)
    nok_list = [r for r in results if r["result"] == "NOK"]

    # ── NOK SUMMARY ───────────────────────────────────────────
    print_nok_summary(nok_list)

    # ── FINAL SUMMARY ─────────────────────────────────────────
    print("\n" + "═" * 50)
    print(f"  INSPECTION COMPLETE")
    print(f"  Date     : {date_str}")
    print(f"  Duration : {round(duration, 1)}s")
    print(f"  Images   : {report['total_images']} / {TOTAL_VEINS * 2}")
    print(f"  OK       : {report['ok']}")
    print(f"  NOK      : {report['nok']}")
    print(f"  Session  : {session_dir}")
    print("═" * 50)

    # ── BEEP ──────────────────────────────────────────────────
    subprocess.run(["paplay", "/usr/share/sounds/freedesktop/stereo/complete.oga"],capture_output=True)

    return report

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(IMAGES_LOCAL, exist_ok=True)

    ser  = stm32_connect()
    sock = rpi5_connect()

    # ── Manual positioning ────────────────────────────────────
    manual_positioning(ser)

    print("\n" + "═" * 50)
    print("  SYSTEM READY")
    print("  → Camera faces LEFT wall ?")
    print("  → Lighting connected ?")
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