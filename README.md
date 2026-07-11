# Aegis: Distributed Edge-AI Video Analytics Pipeline

Aegis is a containerized, high-velocity microservices pipeline designed for real-time video streaming and low-latency computer vision inference at the edge. 

Instead of deploying a monolithic machine learning script, this repository splits the architecture into independent, highly decoupled Docker microservices orchestrated via Docker Compose. The compute worker leverages native NVIDIA GPU passthrough acceleration to run real-time tracking using an attention-centric YOLOv12 model architecture.

---

## 🏗️ System Architecture & Data Flow

The pipeline bypasses single-point-of-failure blocks by executing workflows asynchronously through an in-memory database shock absorber:

1. **`ingestion-service` (OpenCV):** Captures high-definition camera streams or video files, processes frames natively to standard inference sizes (640x640), compresses the matrices into lightweight JPEG byte structures, and streams them out.
2. **`broker-service` (Redis):** Acts as a high-speed, memory-bounded (`maxlen=100`) transaction buffer running completely in system RAM, preventing pipeline ingestion lag or memory leaks.
3. **`inference-service` (YOLOv12 + PyTorch):** A heavy compute consumer service that subscribes to the frame broker, maps data back to tensor streams, processes object arrays via YOLOv12 with host GPU virtualization, and posts spatial coordinates.
4. **`dashboard-service` (Streamlit):** A front-facing dashboard that reads asynchronous video vectors and bounding box structures, composites telemetries reactively, and graphs inference performance.

---

## 🛠️ Tech Stack & Systems Engineering

* **Infrastructure & Containerization:** Docker, Docker Compose (Multi-stage builds, Bridge Networking).
* **Hardware Virtualization:** NVIDIA Container Toolkit (Host GPU Passthrough / CUDA runtime hooks).
* **Data Layer & Messaging:** Redis Streams (Pub/Sub message queuing, bounded ring-buffers).
* **Computer Vision & Core AI:** PyTorch, YOLOv12 (Attention-centric real-time object detection), OpenCV.
* **User Interface:** Streamlit (Reactive data components, real-time video rendering loop).

---

## 📂 Repository Layout

```text
edge-vision-pipeline/
├── deployments/
│   └── docker-compose.yml     # Master container network manifest
├── services/
│   ├── ingestion/             # Video capture & serialization engine
│   │   ├── src/main.py
│   │   └── Dockerfile
│   ├── inference/             # GPU-accelerated YOLOv12 worker
│   │   ├── src/main.py
│   │   └── Dockerfile
│   └── dashboard/             # Front-facing reactive UI telemetry
│       ├── src/app.py
│       └── Dockerfile
└── .gitignore
