Spatio-Temporal Movement Anomaly Detection System

B.Tech Final Year Project — Defense & Surveillance Analytics (DRDO-Oriented)

##Technical Description

This project presents a **Spatio-Temporal Movement Anomaly Detection System** designed for defense and surveillance applications. The system ingests **SAR (Synthetic Aperture Radar)** and **Optical satellite imagery**, performing multi-modal analysis to detect, track, and classify anomalous movement patterns of ground, aerial, and maritime targets.

Unlike conventional object detection pipelines that focus solely on *what* is present in a scene, this system emphasizes **movement anomaly detection** — analyzing *how* objects move over time and space to identify suspicious behaviors such as evasive maneuvers, unauthorized convergence, loitering in sensitive zones, and coordinated formations.

## Core Components

### 1. YOLOv8-OBB (Oriented Object Detection)
The detection module uses **YOLOv8 with Oriented Bounding Box (OBB)** support, enabling precise localization of rotated objects common in overhead satellite imagery. The OBB head outputs five parameters `(x, y, w, h, θ)` per detection, crucial for vehicles and structures that appear at arbitrary orientations. Eight military-relevant classes are supported: vehicle, aircraft, ship, personnel, radar installation, missile launcher, armored vehicle, and helicopter.

### 2. ByteTrack (Multi-Object Tracking)
**ByteTrack** provides robust multi-object tracking through its innovative two-stage association strategy:
- **Stage 1:** High-confidence detections are matched to existing tracks using IoU-based cost matrices
- **Stage 2:** Low-confidence detections are re-associated with unmatched tracks, recovering occluded or partially visible targets

A Kalman filter estimates the state vector `[cx, cy, w, h, vx, vy, vw, vh]`, providing smooth velocity estimation for downstream temporal analysis.

### 3. Temporal Transformer (Spatio-Temporal Reasoning)
A custom **Transformer encoder** captures long-range temporal dependencies across trajectory sequences:
- **Input Features:** 7-dimensional vectors `[cx, cy, w, h, angle, vx, vy]` per timestep
- **Architecture:** 4-layer encoder with 8 attention heads, 128-dim model, sinusoidal positional encoding
- **Output:** Per-timestep motion embeddings and sequence-level anomaly logits
- The self-attention mechanism enables the model to correlate motion patterns across distant time steps, identifying behaviors that rule-based systems might miss.

### 4. Multi-Indicator Risk Scoring
Five anomaly indicators are fused with learned weights:
| Indicator | Weight | Description |
|-----------|--------|-------------|
| Speed Anomaly | 20% | Sudden acceleration/deceleration (σ-threshold) |
| Direction Anomaly | 20% | Abrupt heading changes, zigzag evasion |
| Loitering | 15% | Circular/stationary patterns in areas of interest |
| Convergence | 15% | Multiple objects converging to a common point |
| Transformer Score | 30% | Learned temporal pattern deviation |

Risk levels: **LOW** (0–0.3), **MEDIUM** (0.3–0.6), **HIGH** (0.6–0.85), **CRITICAL** (0.85–1.0)

### 5. Spatial Anomaly Map
A 32×32 grid heatmap accumulates risk scores from tracked trajectories with Gaussian spatial spreading, producing a visual overview of threat concentration zones. Hotspot extraction identifies regions exceeding an intensity threshold.

---

## Project Structure
stad_project/
│
├── data/
│   ├── sar/                     # SAR satellite imagery
│   ├── optical/                 # Optical satellite imagery
│   └── sequences/               # Temporal sequences
│
├── models/
│   ├── __init__.py
│   ├── detector.py              # YOLOv8-OBB wrapper
│   ├── tracker.py               # ByteTrack integration
│   ├── transformer.py           # Temporal Transformer model
│   └── anomaly.py               # Risk scoring & anomaly detection
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py         # SAR filtering, fusion, synthetic data
│   ├── visualization.py         # Heatmaps, dashboards, overlay rendering
│   └── config.py                # Hyperparameters & system configuration
│
├── outputs/                     # Generated visualizations & reports
├── weights/                     # Model weights directory
├── main.py                      # Pipeline runner
├── requirements.txt
└── README.md
---
## Defense Readiness (DRDO Orientation)

This system is designed with defense and surveillance applications in mind:
- **Multi-modal fusion:** Combines SAR (all-weather, day/night) with Optical imagery for robust detection
- **Military object classes:** Detects vehicles, aircraft, ships, missile launchers, radar installations
- **Anomaly focus:** Goes beyond detection to identify *behavioral* threats — convergence, evasion, loitering
- **Real-time capable:** Modular pipeline architecture supports GPU acceleration for operational deployment
- **Structured reporting:** JSON-based threat reports for integration with C4ISR systems
- **Spatial awareness:** Anomaly heatmaps enable rapid situational assessment for command centers

---

## Technical Specifications

| Component | Specification |
|-----------|--------------|
| Detection Model | YOLOv8-OBB (anchor-free, CSPDarknet-53) |
| Tracker | ByteTrack with Kalman Filter |
| Temporal Model | Transformer (4L encoder, 8 heads, d=128) |
| Input Modalities | SAR + Optical satellite imagery |
| OBB Output | (x, y, w, h, θ) oriented bounding boxes |
| Risk Indicators | 5 (speed, direction, loitering, convergence, transformer) |
| Anomaly Map | 32×32 spatial grid with Gaussian spreading |
| Framework | PyTorch + OpenCV |

---

## Authors
CM_06
