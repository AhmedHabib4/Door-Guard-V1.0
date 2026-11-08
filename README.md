<div align="center">

# Door Guard — AI Face Recognition

**A Complete End-to-End Real-Time Home Security System on CPU-Only Hardware**

*Continuous RTSP stream processing, custom dataset generation, and ArcFace multi-frame voting logic.*

<br>

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![CV](https://img.shields.io/badge/CV-OpenCV-green?logo=opencv&logoColor=white)
![ML](https://img.shields.io/badge/ML-YOLOv8-orange)
![Framework](https://img.shields.io/badge/framework-PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Database](https://img.shields.io/badge/database-SQLite-003B57?logo=sqlite&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-brightgreen)
![Status](https://img.shields.io/badge/status-portfolio_project-E05D44)

---
</div>

## 📖 Overview
**Door Guard V1.0** is a complete, self-hosted, CPU-optimized smart security pipeline designed to run 24/7 on older hardware (e.g., an old laptop). It processes multiple RTSP camera streams in near real-time, records continuous footage, and utilizes a cascade of AI models to detect motion, identify vehicles, and recognize faces. 

Unlike off-the-shelf cloud solutions, this project ensures 100% data privacy by keeping all processing and storage local. It features a custom-trained ArcFace PyTorch adapter built from months of security footage, significantly reducing False Reject Rates (FRR) while maintaining a strict <1% False Accept Rate (FAR) for unknown visitors. 

## ✨ Key Features
*   **Dual-Stream Processing:** Threaded camera readers handle both high-res (for FFmpeg recording) and low-res (for AI inference) RTSP streams seamlessly.
*   **CPU-Optimized AI Cascade:** Utilizes a lightweight motion detector gate to keep heavy neural networks idle 90% of the time, allowing smooth 3-5 FPS processing on low-end CPUs.
*   **Robust Face Recognition:** Uses InsightFace (MobileFaceNet) for baseline embeddings, enhanced by a custom-trained PyTorch ArcFace adapter to distinguish family members from strangers under various door conditions (e.g., infrared, odd angles).
*   **Smart Rule Engine:** Multi-frame identity voting combined with customized event rules (e.g., knocking, entering, unusual hours).
*   **Real-Time Alerts:** Asynchronous Telegram bot integration delivers instant push notifications with snapshots for known arrivals and unknown visitors.
*   **Local Web Dashboard:** FastAPI-powered dashboard to monitor the SQLite event logs and snapshots.
*   **Active Learning Loop:** Automatically saves borderline/unknown face crops for continuous model retraining.

## 🛠️ Tech Stack
*   **Language:** Python 3.11+
*   **Computer Vision:** OpenCV, FFmpeg
*   **Deep Learning / AI:** PyTorch, Ultralytics YOLOv8 (ONNX exported), InsightFace (`buffalo_sc`)
*   **Database:** SQLite3
*   **Web / API:** FastAPI, Uvicorn, Requests
*   **Deployment:** Ubuntu systemd, Docker (optional)

## 🚀 Highlights (Portfolio Notes)
*   **Edge AI Deployment:** Successfully running heavy vision pipelines on an old i5 CPU without thermal throttling.
*   **Semi-Automatic Labeling:** Implemented DBSCAN clustering on facial embeddings to rapidly label thousands of faces from raw footage.
*   **Open-Set Evaluation:** Rigorously tested using standard biometrics metrics (FAR vs. FRR thresholds) rather than simple closed-set accuracy.
*   **Data Pipeline:** Engineered a complete data collection, extraction, alignment, augmentation, and training pipeline entirely from scratch.

## 🏗️ Architecture
1.  **Ingestion:** RTSP streams are read asynchronously. High-res streams are piped directly to FFmpeg (segmenting every 10 mins).
2.  **Motion Gate:** Background subtraction filters out static frames.
3.  **Object Detection:** YOLOv8 finds people and vehicles.
4.  **Face Analysis:** InsightFace extracts and aligns faces, outputting a 512-D embedding.
5.  **Matching & Tracking:** Centroid tracking tracks subjects across frames; a trained ArcFace adapter matches embeddings against known prototypes.
6.  **Action:** The Rule Engine determines if an event is notable, logs it to SQLite, and pushes to a Telegram queue.

## 📂 Repository Structure
```text
door_guard_ai/
├── config.yaml             # Camera URLs, zones, thresholds, and rules
├── .env                    # Secrets: Telegram token/chat ID
├── dataset/                # Baseline photo enrollment directories
├── db/                     # SQLite database (door.sqlite)
├── docs/                   # DATA_CARD.md and MODEL_CARD.md
├── eval/                   # Test datasets and evaluation results/ROC plots
├── recordings/             # 24/7 continuous FFmpeg video segments
├── snapshots/              # Event image captures
├── src/
│   ├── camera.py           # Threaded RTSP reader with auto-reconnect
│   ├── dashboard.py        # FastAPI web interface
│   ├── database.py         # SQLite schema and helpers
│   ├── detector.py         # YOLOv8 object detection wrapper
│   ├── faces.py            # InsightFace wrapper and quality filters
│   ├── main.py             # Core real-time video pipeline
│   ├── matcher.py          # Face matching and multi-frame voting logic
│   ├── motion.py           # Background subtraction CPU saver
│   ├── notifier.py         # Async Telegram alerting thread
│   ├── tracker.py          # Centroid tracking
│   └── train/              # Offline dataset creation & PyTorch training scripts
└── tests/                  # Unit tests for core logic
```

## ⚡ Quick Start
1. **Clone & Install:**
   ```bash
   git clone https://github.com/AhmedHabib4/Door-Guard-V1.0.git
   cd Door-Guard-V1.0
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Configure Secrets:** Create a `.env` file with `TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID`.
3. **Configure Cameras:** Edit `config.yaml` with your local RTSP credentials and bounding lines.
4. **Enroll Baseline:** Place photos in `dataset/` (e.g., `dataset/owner_Ahmed`) and run:
   ```bash
   python src/enroll.py
   ```
5. **Run the Pipeline:**
   ```bash
   python src/main.py
   ```
6. **Launch Dashboard (Optional):**
   ```bash
   uvicorn src.dashboard:app --host 0.0.0.0 --port 8000
   ```

## 📚 Documentation
For deeper technical insights into how the custom datasets were built and how the PyTorch models were trained, please review:
*   [Data Card](docs/DATA_CARD.md): Information on video splits, labeling methodology, and privacy.
*   [Model Card](docs/MODEL_CARD.md): Metrics, FRR/FAR trade-offs, and known edge-case failures.

## 🧪 Testing
Unit tests and evaluation scripts are provided to ensure pipeline integrity and model accuracy:
*   Run unit tests (tracker, matcher, line crossing): `pytest tests/`
*   Evaluate open-set facial recognition accuracy: `python eval/eval_faces.py`
*   Verify camera connectivity: `python src/test_camera.py`

## Author

**Ahmed Said Habib** — [@AhmedHabib4](https://github.com/AhmedHabib4) · portfolio project.

Licensed under the [MIT License](https://github.com/AhmedHabib4/Door-Guard-V1.0/blob/main/LICENSE).
