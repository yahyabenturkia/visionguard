#!/usr/bin/env python3
"""
VisionGuard — RPi5 TCP Server
Image acquisition server for dataset collection.
Runs on RPi5 (192.168.10.2), listens for commands from laptop.
"""

import os
os.environ["DISPLAY"] = ":0"
os.environ["LIBCAMERA_LOG_LEVELS"] = "4"

import socket
import shutil
import json
import time
from datetime import datetime
from picamera2 import Picamera2, Preview

# ─── Configuration ────────────────────────────────────────────────────────────
HOST        = "0.0.0.0"
PORT        = 9999
DATASET_DIR = "/home/pi5/dataset"
GOOD_DIR    = os.path.join(DATASET_DIR, "good")
DEFECT_DIR  = os.path.join(DATASET_DIR, "defect")
INSPECT_DIR = "/home/pi5/inspection"
RESOLUTION  = (1456, 1088)

# ─── Helpers ──────────────────────────────────────────────────────────────────
def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:22]

def disk_space_bar():
    stat = shutil.disk_usage(DATASET_DIR)
    percent = stat.used / stat.total * 100
    free_gb = stat.free / (1024**3)
    filled = int(percent / 5)
    bar = "█" * filled + "░" * (20 - filled)
    return f"[{bar}] {percent:.1f}% used ({free_gb:.1f} GB free)"

def save_session_log(counts):
    log = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M:%S"),
        "good_count": counts["good"],
        "defect_count": counts["defect"],
        "total": counts["good"] + counts["defect"],
    }
    log_path = os.path.join(DATASET_DIR, f"session_{timestamp()}.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  [LOG] Session log saved: {log_path}")

# ─── Camera ───────────────────────────────────────────────────────────────────
def init_camera():
    print("  [CAM] Initializing camera...", end="", flush=True)
    cam = Picamera2()
    config = cam.create_still_configuration(
        main={"size": RESOLUTION, "format": "XBGR8888"},
        display="main"
    )
    cam.configure(config)
    cam.start_preview(Preview.QTGL, x=0, y=0, width=800, height=600)
    cam.start()
    time.sleep(3)
    print(" OK")
    return cam

def capture_image(cam, label, wall):
    if label == "good":
        save_dir = GOOD_DIR
    elif label == "defect":
        save_dir = DEFECT_DIR
    else:
        save_dir = INSPECT_DIR
    filename = f"{label}_{wall}_{timestamp()}.jpg"
    temp_path = f"/tmp/{filename}"
    final_path = os.path.join(save_dir, filename)
    cam.capture_file(temp_path)
    shutil.move(temp_path, final_path)
    return filename, final_path

# ─── Command Handler ──────────────────────────────────────────────────────────
def handle_command(cmd, cam, counts):
    cmd = cmd.strip()

    if cmd.startswith("CAPTURE:"):
        parts = cmd.split(":")
        if len(parts) != 3:
            return "ERR:invalid_format", False
        label = parts[1].strip().lower()
        wall = parts[2].strip().upper()
        if label not in ("good", "defect" , "inspect"):
            return "ERR:invalid_label", False
        if wall not in ("R", "L"):
            return "ERR:invalid_wall", False
        try:
            filename, path = capture_image(cam, label, wall)
            counts[label] += 1
            disk = disk_space_bar()
            print(f"  [CAP] {label.upper()} {wall} → {filename} | {disk}")
            return f"OK:{filename}", False
        except Exception as e:
            print(f"  [ERR] Capture failed: {e}")
            return "ERR:capture_failed", False

    elif cmd.startswith("DELETE:"):
        filename = cmd.split(":", 1)[1].strip()
        for folder in (GOOD_DIR, DEFECT_DIR):
            filepath = os.path.join(folder, filename)
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"  [DEL] Deleted: {filename}")
                return "OK", False
        print(f"  [DEL] File not found: {filename}")
        return "ERR:not_found", False

    elif cmd == "QUIT":
        return "BYE", True

    else:
        return "ERR:unknown_command", False

# ─── Session Handler ──────────────────────────────────────────────────────────
def handle_session(conn, cam):
    counts = {"good": 0, "defect": 0, "inspect": 0}
    buffer = ""
    session_active = True

    try:
        while session_active:
            data = conn.recv(1024).decode("utf-8")
            if not data:
                print("  [NET] Connection closed by laptop.")
                break

            buffer += data
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue

                print(f"  [CMD] Received: {line}")
                response, quit_flag = handle_command(line, cam, counts)
                conn.sendall((response + "\n").encode("utf-8"))
                print(f"  [RSP] Sent: {response}")

                if quit_flag:
                    session_active = False
                    break

    except Exception as e:
        print(f"  [ERR] Session error: {e}")
    finally:
        save_session_log(counts)
        print(f"  [SES] Session ended — Good: {counts['good']} | Defect: {counts['defect']}")
        conn.close()

# ─── Main Server ──────────────────────────────────────────────────────────────
def main():
    os.makedirs(GOOD_DIR, exist_ok=True)
    os.makedirs(DEFECT_DIR, exist_ok=True)
    os.makedirs(INSPECT_DIR, exist_ok=True)

    print("\n" + "═" * 50)
    print("   VISIONGUARD — RPi5 IMAGE SERVER")
    print("   Pursuit Aerospace Tunisia")
    print("═" * 50)

    cam = init_camera()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    print(f"\n  [NET] Server listening on port {PORT}")
    print("  [NET] Waiting for laptop connection...")

    try:
        while True:
            conn, addr = server.accept()
            print(f"  [NET] Laptop connected from {addr[0]}")
            handle_session(conn, cam)
            print("  [NET] Waiting for next connection...")

    except KeyboardInterrupt:
        print("\n  [SYS] Server interrupted.")
    finally:
        cam.stop_preview()
        cam.stop()
        server.close()
        print("  [SYS] Camera and server closed cleanly.")

if __name__ == "__main__":
    main()