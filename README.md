# VisionGuard — Automated Inspection System

PFE Project — ENISo 2026  
Student: Yahya Benturkia  
Host Company: Pursuit Aerospace Tunisia  

## Overview
Automated visual inspection system for ceramic core M00.  
Detects surface defects (pores/pits ≥ 0.8mm) on 80 internal vein walls.  
Target cycle time: ≤ 3 minutes/part.

## System Architecture
- **Laptop** — Master orchestration (Python)
- **STM32 Nucleo F446RE** — Motion control (C, UART slave)
- **Raspberry Pi 5** — Image acquisition (Python, CSI camera)

## Repository Structure
- `firmware/` — STM32 motion control firmware
- `vision/` — RPi5 image acquisition scripts
- `orchestration/` — Laptop master script
- `ai/` — Inference pipeline
- `tests/` — Validation and test scripts
- `docs/` — Project assets and diagrams

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
