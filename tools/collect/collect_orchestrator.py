#!/usr/bin/env python3
"""
VisionGuard — Dataset Collection Orchestrator
Runs on laptop. Controls STM32 (motion + lighting) + RPi5 (capture).
"""

import os
import sys
import tty
import termios
import serial
import socket
import time
import subprocess
from datetime import datetime

# ─── Configuration ────────────────────────────────────────────────────────────
STM32_PORT    = "/dev/ttyACM0"
STM32_BAUD    = 115200
RPI5_HOST     = "192.168.10.2"
RPI5_PORT     = 9999
DATASET_LOCAL = os.path.expanduser("~/visionguard/raw_data")

SPEED_MIN     = 100
SPEED_MAX     = 2000
SPEED_DEFAULT = 800
SPEED_STEP    = 200
TOTAL_VEINS   = 40

# Lighting: 10 levels (0-9). Duty range 0-999 on STM32 PWM.
LIGHT_MAX_DUTY = 999
LIGHT_STEP     = LIGHT_MAX_DUTY // 9   # 111 per level → 9*111 = 999

def level_to_duty(level):
    return min(level * LIGHT_STEP, LIGHT_MAX_DUTY)

# ─── Terminal helpers ─────────────────────────────────────────────────────────
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

def clear_line():
    print("\r" + " " * 80 + "\r", end="", flush=True)

# ─── Display ──────────────────────────────────────────────────────────────────
def print_display(state):
    os.system("clear")
    light_level = state["light_level"]
    light_str = "OFF" if light_level == 0 else f"{light_level}/9 ({level_to_duty(light_level)})"
    print("═" * 51)
    print("   VISIONGUARD — DATASET COLLECTION")
    print("   Pursuit Aerospace Tunisia")
    print("═" * 51)
    print(f"  Position : Vein {state['vein']:02d} / {TOTAL_VEINS}")
    print(f"  Speed    : {state['speed']} steps/sec")
    print(f"  Light    : {light_str}")
    print(f"  Label    : {'✔ GOOD' if state['label'] == 'good' else '✘ DEFECT'}")
    print(f"  Wall     : {'RIGHT' if state['wall'] == 'R' else 'LEFT'}")
    print(f"  Captured : Good: {state['counts']['good']} | Defect: {state['counts']['defect']} | Total: {state['counts']['good'] + state['counts']['defect']}")
    if state['last_file']:
        print(f"  Last     : {state['last_file']}")
    else:
        print(f"  Last     : —")
    print("─" * 51)
    print("  H=Home  ←→=Fine(1°)  N/P=Vein  ↑↓=Speed")
    print("  0-9=Light(0=off)  G=Good  B=Defect")
    print("  R=Right  L=Left  ENTER=Capture  D=Del  Q=Quit")
    print("═" * 51)

# ─── STM32 Communication ──────────────────────────────────────────────────────
def stm32_send(ser, cmd):
    ser.write((cmd + "\n").encode())
    response = ser.readline().decode().strip()
    return response

def stm32_connect():
    print("  [STM32] Connecting...", end="", flush=True)
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
    print(" OK (no READY — firmware already running)")
    return ser

# ─── RPi5 Communication ───────────────────────────────────────────────────────
def rpi5_connect():
    print("  [RPi5] Connecting...", end="", flush=True)
    sock = socket.socket()
    sock.connect((RPI5_HOST, RPI5_PORT))
    sock.settimeout(10)
    print(" OK")
    return sock

def rpi5_send(sock, cmd):
    sock.sendall((cmd + "\n").encode())
    response = sock.recv(1024).decode().strip()
    return response

# ─── SCP Transfer ─────────────────────────────────────────────────────────────
def transfer_dataset():
    print("\n  [SCP] Transferring dataset from RPi5...")
    os.makedirs(DATASET_LOCAL, exist_ok=True)
    cmd = [
        "scp", "-r",
        f"pi5@{RPI5_HOST}:/home/pi5/dataset/good",
        f"pi5@{RPI5_HOST}:/home/pi5/dataset/defect",
        DATASET_LOCAL
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("  [SCP] Transfer complete.")
        # Clean up RPi5 after successful transfer
        cleanup = subprocess.run([
            "ssh", f"pi5@{RPI5_HOST}",
            "rm -rf /home/pi5/dataset/good/* /home/pi5/dataset/defect/*"
        ], capture_output=True, text=True)
        if cleanup.returncode == 0:
            print("  [SCP] RPi5 dataset cleared.")
        else:
            print(f"  [SCP] Warning: RPi5 cleanup failed: {cleanup.stderr}")
    else:
        print(f"  [SCP] Transfer failed: {result.stderr}")

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(DATASET_LOCAL, exist_ok=True)

    print("\n" + "═" * 51)
    print("   VISIONGUARD — DATASET COLLECTION")
    print("   Startup...")
    print("═" * 51)

    ser = stm32_connect()
    sock = rpi5_connect()

    state = {
        "vein": 0,
        "speed": SPEED_DEFAULT,
        "label": "good",
        "wall": "R",
        "last_file": None,
        "counts": {"good": 0, "defect": 0},
        "homed": False,
        "light_level": 0,
        "message": "Press H to home before starting."
    }

    print_display(state)

    try:
        while True:
            key = get_keypress()

            # ── HOME ──────────────────────────────────────────────
            if key.lower() == 'h':
                r = stm32_send(ser, "HOME")
                if r == "OK":
                    state["vein"] = 1
                    state["homed"] = True
                    state["message"] = "Homed. Position set to vein 1."
                else:
                    state["message"] = f"HOME failed: {r}"

            # ── FINE RIGHT ────────────────────────────────────────
            #elif key == '\x1b[C':
            #    r = stm32_send(ser, "MOV:-27")
            #    state["message"] = f"→ Fine +1°  [{r}]"
            
            # ── FINE LEFT ─────────────────────────────────────────
            #elif key == '\x1b[D':
            #    r = stm32_send(ser, "MOV:27")
            #    state["message"] = f"← Fine -1°  [{r}]"
            # ── FINE RIGHT ────────────────────────────────────────
            elif key == '\x1b[C':
                r = stm32_send(ser, "MOV:-27")
                state["total_steps"] = 0
                state["message"] = f"→ Fine +1°  [{r}]"

            # ── FINE LEFT ─────────────────────────────────────────
            elif key == '\x1b[D':
                r = stm32_send(ser, "MOV:27")
                state["total_steps"] = 0
                state["message"] = f"← Fine -1°  [{r}]"
            # ── SPEED UP ──────────────────────────────────────────
            elif key == '\x1b[A':
                new_speed = min(state["speed"] + SPEED_STEP, SPEED_MAX)
                r = stm32_send(ser, f"SPEED:{new_speed}")
                if r == "OK":
                    state["speed"] = new_speed
                state["message"] = f"↑ Speed: {state['speed']} steps/sec  [{r}]"

            # ── SPEED DOWN ────────────────────────────────────────
            elif key == '\x1b[B':
                new_speed = max(state["speed"] - SPEED_STEP, SPEED_MIN)
                r = stm32_send(ser, f"SPEED:{new_speed}")
                if r == "OK":
                    state["speed"] = new_speed
                state["message"] = f"↓ Speed: {state['speed']} steps/sec  [{r}]"

            # ── NEXT VEIN ─────────────────────────────────────────
            #elif key.lower() == 'n':
            #    r = stm32_send(ser, "MOV:120")
            #    if r == "OK":
            #        state["vein"] = min(state["vein"] + 1, TOTAL_VEINS)
            #    state["message"] = f"N → Next vein  [{r}]"
            
            # ── PREVIOUS VEIN ─────────────────────────────────────
            #elif key.lower() == 'p':
            #    r = stm32_send(ser, "MOV:-120")
            #    if r == "OK":
            #        state["vein"] = max(state["vein"] - 1, 1)
            #    state["message"] = f"P → Previous vein  [{r}]"
            #------- next and previous handler -----
            elif key.lower() == 'n':
                state["total_steps"] = state.get("total_steps", 0) + 1
                step = 124 if state["total_steps"] % 2 == 0 else 120
                r = stm32_send(ser, f"MOV:{step}")
                if r == "OK":
                    state["vein"] = min(state["vein"] + 1, TOTAL_VEINS)
                state["message"] = f"N → Next vein (MOV:{step})  [{r}]"

            elif key.lower() == 'p':
                state["total_steps"] = state.get("total_steps", 0) - 1
                step = 124 if state["total_steps"] % 2 == 0 else 120
                r = stm32_send(ser, f"MOV:-{step}")
                if r == "OK":
                    state["vein"] = max(state["vein"] - 1, 1)
                state["message"] = f"P → Previous vein (MOV:-{step})  [{r}]"
            # ── LIGHT LEVEL 0-9 ───────────────────────────────────
            elif key in '0123456789':
                level = int(key)
                duty = level_to_duty(level)
                r = stm32_send(ser, f"LIGHT:SET:{duty}")
                if r == "OK":
                    state["light_level"] = level
                    if level == 0:
                        state["message"] = f"💡 Light OFF  [{r}]"
                    else:
                        state["message"] = f"💡 Light level {level}/9 (duty {duty})  [{r}]"
                else:
                    state["message"] = f"Light failed: {r}"

            # ── LABEL GOOD ────────────────────────────────────────
            elif key.lower() == 'g':
                state["label"] = "good"
                state["message"] = "Label set to GOOD."

            # ── LABEL DEFECT ──────────────────────────────────────
            elif key.lower() == 'b':
                state["label"] = "defect"
                state["message"] = "Label set to DEFECT."

            # ── WALL RIGHT ────────────────────────────────────────
            elif key.lower() == 'r':
                state["wall"] = "R"
                state["message"] = "Wall set to RIGHT."

            # ── WALL LEFT ─────────────────────────────────────────
            elif key.lower() == 'l':
                state["wall"] = "L"
                state["message"] = "Wall set to LEFT."

            # ── CAPTURE ───────────────────────────────────────────
            elif key in ('\r', '\n', ' '):
                r = rpi5_send(sock, f"CAPTURE:{state['label']}:{state['wall']}")
                if r.startswith("OK:"):
                    filename = r.split(":", 1)[1]
                    state["last_file"] = filename
                    state["counts"][state["label"]] += 1
                    state["message"] = f"✔ Captured: {filename}"
                else:
                    state["message"] = f"✘ Capture failed: {r}"

            # ── DELETE LAST ───────────────────────────────────────
            elif key.lower() == 'd':
                if state["last_file"]:
                    r = rpi5_send(sock, f"DELETE:{state['last_file']}")
                    if r == "OK":
                        label = "good" if "good" in state["last_file"] else "defect"
                        state["counts"][label] = max(0, state["counts"][label] - 1)
                        state["message"] = f"✘ Deleted: {state['last_file']}"
                        state["last_file"] = None
                    else:
                        state["message"] = f"Delete failed: {r}"
                else:
                    state["message"] = "Nothing to delete."

            # ── QUIT ──────────────────────────────────────────────
            elif key.lower() == 'q':
                stm32_send(ser, "LIGHT:OFF")
                state["light_level"] = 0
                r = rpi5_send(sock, "QUIT")
                state["message"] = "Session ended. Transferring dataset..."
                print_display(state)
                transfer_dataset()
                break

            # ── CTRL+C ────────────────────────────────────────────
            elif key == '\x03':
                stm32_send(ser, "LIGHT:OFF")
                break

            print_display(state)
            print(f"  {state['message']}")

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        stm32_send(ser, "LIGHT:OFF")
        ser.close()
        sock.close()
        total = state["counts"]["good"] + state["counts"]["defect"]
        print(f"\n  Session summary — Good: {state['counts']['good']} | Defect: {state['counts']['defect']} | Total: {total}")
        print("  Goodbye.")

if __name__ == "__main__":
    main()