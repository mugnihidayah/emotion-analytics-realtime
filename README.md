# 🧠 Real-time Emotion Analytics Dashboard

A production-grade, deep learning-powered system for **real-time facial emotion recognition** and **engagement analytics**. Built with a decoupled architecture — a premium **Next.js** frontend and a high-performance **FastAPI** backend — designed for deployment on **Vercel** + **Hugging Face Spaces**.

![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=next.js&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178c6?logo=typescript&logoColor=white)
![WebSocket](https://img.shields.io/badge/WebSocket-Real--time-blueviolet)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Deployment](#deployment)
- [Model Details](#model-details)
- [Training](#training)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project goes far beyond basic emotion classification. It delivers a **full engagement analytics pipeline** — detecting facial expressions in real-time, computing weighted engagement scores, and visualizing trends through a premium glassmorphism dashboard.

The system uses a **dual-model inference pipeline**:
1. **YOLOv8** for robust face detection
2. **Custom ResNet-SE CNN** for 7-class emotion classification

All inference runs server-side via a **WebSocket connection**, while the browser handles camera capture and renders all visualizations client-side — achieving low latency with minimal infrastructure requirements.

---

## Key Features

| Feature | Description |
|---------|-------------|
| 🎯 **Real-time Inference** | Browser captures frames → WebSocket → YOLO + CNN → JSON response at ~8 FPS |
| 📊 **Engagement Scoring** | Dynamic 0-100% score based on weighted emotion probabilities |
| 📈 **Live Trend Chart** | Canvas-based sparkline with gradient fill showing engagement over time |
| 🎨 **Premium UI** | Dark glassmorphism theme with micro-animations and gradient accents |
| 🔲 **Smart Bounding Box** | Corner-style face detection overlay with color-coded engagement |
| 📉 **Probability Distribution** | Real-time probability bars for all 7 emotion classes |
| 🔄 **EMA Smoothing** | Exponential Moving Average eliminates prediction jitter |
| 🔌 **Auto-Reconnect** | WebSocket with exponential backoff for resilient connections |
| 📱 **Responsive Design** | Adapts seamlessly from desktop to mobile viewports |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        USER'S BROWSER                                │
│                                                                      │
│   ┌─────────────┐    ┌──────────────┐    ┌────────────────────┐     │
│   │   Camera     │───▶│ Frame Capture│───▶│  WebSocket Client  │     │
│   │ getUserMedia │    │ Canvas @8FPS │    │  (base64 JPEG)     │     │
│   └─────────────┘    └──────────────┘    └─────────┬──────────┘     │
│                                                     │                │
│   ┌─────────────┐    ┌──────────────┐    ┌─────────▼──────────┐     │
│   │  Sparkline   │    │  Engagement  │    │  Emotion Display   │     │
│   │  Trend Chart │    │  Score Bar   │    │  + BBox Overlay    │     │
│   └─────────────┘    └──────────────┘    └────────────────────┘     │
│                                                                      │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │ WebSocket (wss://)
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     FASTAPI BACKEND (HF Spaces)                      │
│                                                                      │
│   ┌─────────────┐    ┌──────────────┐    ┌────────────────────┐     │
│   │  Base64      │───▶│   YOLOv8     │───▶│  Face ROI          │     │
│   │  Decode      │    │   Detection  │    │  Extraction        │     │
│   └─────────────┘    └──────────────┘    └─────────┬──────────┘     │
│                                                     │                │
│   ┌─────────────┐    ┌──────────────┐    ┌─────────▼──────────┐     │
│   │  JSON        │◀───│  Engagement  │◀───│  ResNet-SE CNN     │     │
│   │  Response    │    │  Calculator  │    │  Classifier        │     │
│   └─────────────┘    └──────────────┘    └────────────────────┘     │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Camera Capture**: Browser accesses webcam via `getUserMedia` API
2. **Frame Extraction**: Canvas captures video frames at 8 FPS, encodes as base64 JPEG
3. **WebSocket Transport**: Frames sent to FastAPI backend over persistent WebSocket
4. **Face Detection**: YOLOv8 locates faces and extracts bounding boxes
5. **Emotion Classification**: Face ROI preprocessed (grayscale, 48×48) and classified by CNN
6. **Temporal Smoothing**: 5-frame moving average eliminates prediction noise
7. **Engagement Scoring**: Weighted sum of emotion probabilities → 0-100% score
8. **JSON Response**: `{emotion, score, bbox, probabilities}` sent back to client
9. **Visualization**: Frontend renders bounding box overlay, engagement bar, sparkline, and probability bars

---

## Tech Stack

### Frontend
| Technology | Purpose |
|-----------|---------|
| **Next.js 16** | React framework with App Router |
| **TypeScript** | Type-safe component development |
| **Canvas API** | Bounding box overlay & sparkline rendering |
| **WebSocket** | Real-time bidirectional communication |
| **CSS3** | Glassmorphism design system with animations |

### Backend
| Technology | Purpose |
|-----------|---------|
| **FastAPI** | High-performance async Python web framework |
| **PyTorch** | Deep learning inference engine |
| **YOLOv8 (Ultralytics)** | Real-time face detection |
| **OpenCV** | Image preprocessing & decoding |
| **Uvicorn** | ASGI server with WebSocket support |

### Infrastructure
| Technology | Purpose |
|-----------|---------|
| **Vercel** | Frontend hosting (edge CDN) |
| **Hugging Face Spaces** | Backend hosting (Docker SDK) |
| **Docker** | Containerized backend deployment |
| **Docker Compose** | Local multi-service development |

---

## Project Structure

```
emotion-analytics-realtime/
│
├── frontend/                           # Next.js application
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx              # Root layout with SEO metadata
│   │   │   ├── page.tsx                # Main dashboard (orchestrator)
│   │   │   └── globals.css             # Design system & animations
│   │   │
│   │   ├── components/
│   │   │   ├── CameraFeed.tsx          # Camera + canvas bbox overlay
│   │   │   ├── EngagementBar.tsx       # Animated engagement score bar
│   │   │   ├── EmotionDisplay.tsx      # Emoji badge + probability bars
│   │   │   ├── SparklineChart.tsx      # Real-time canvas trend chart
│   │   │   ├── AnalyticsPanel.tsx      # Right-side analytics container
│   │   │   └── StatusIndicator.tsx     # Connection status indicator
│   │   │
│   │   ├── hooks/
│   │   │   ├── useWebSocket.ts         # WebSocket with auto-reconnect
│   │   │   └── useCamera.ts           # Camera access + frame capture
│   │   │
│   │   └── lib/
│   │       └── types.ts                # Shared TypeScript types
│   │
│   ├── .env.example                    # Environment template
│   ├── package.json
│   ├── next.config.ts
│   └── tsconfig.json
│
├── backend/                            # FastAPI server
│   ├── main.py                         # FastAPI app + WebSocket endpoint
│   ├── logic.py                        # EmotionAnalyzer inference engine
│   ├── models/
│   │   ├── best.pt                     # YOLOv8 face detector (~6MB)
│   │   └── model_cnn.pth              # Emotion classifier (~18MB)
│   ├── requirements.txt
│   └── Dockerfile                      # HF Spaces Docker config
│
├── training/                           # Model training notebooks
│   ├── YOLOv8.ipynb                    # YOLOv8 fine-tuning
│   └── classification_training_fer2013.ipynb  # CNN training
│
├── docker-compose.yml                  # Local multi-service setup
├── .gitignore
└── README.md
```

---

## Getting Started

### Prerequisites

- **Node.js** 18+ (frontend)
- **Python** 3.11+ (backend)
- **NVIDIA GPU** with CUDA support (recommended, falls back to CPU)

### 1. Clone the Repository

```bash
git clone https://github.com/mugnihidayah/emotion-analytics-realtime.git
cd emotion-analytics-realtime
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install PyTorch with CUDA (check version with nvidia-smi)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# For CPU only: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
pip install -r requirements.txt

# Run the server
python -m uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```

The backend will start at `http://localhost:7860`. Verify with:
```bash
curl http://localhost:7860/api/health
# Response: {"status":"ready","device":"cuda","labels":["Angry","Disgust",...]}
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Configure backend URL (defaults to localhost:7860)
cp .env.example .env.local

# Run dev server
npm run dev
```

### 4. Open the Dashboard

Navigate to `http://localhost:3000`, click **Start Camera**, grant camera permission, and see real-time emotion analysis!

### Docker (Both Services)

```bash
# Run both services with GPU support
docker compose up --build

# Open http://localhost:3000
```

---

## Deployment

### Frontend → Vercel

1. Push the repository to GitHub
2. Import in [Vercel](https://vercel.com) → set **Root Directory** to `frontend`
3. Add environment variables:
   ```
   NEXT_PUBLIC_WS_URL=wss://YOUR_USERNAME-emotion-api.hf.space/ws/emotion
   NEXT_PUBLIC_API_URL=https://YOUR_USERNAME-emotion-api.hf.space
   ```
4. Deploy!

### Backend → Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
   - Select **Docker** as the SDK
   - Choose hardware (CPU free tier works, GPU for better performance)
2. Push the `backend/` contents:
   ```bash
   cd backend
   git init
   git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/emotion-api
   git add .
   git commit -m "Deploy emotion analytics backend"
   git push origin main
   ```
3. The Space will build automatically using the Dockerfile

---

## Model Details

### Face Detection — YOLOv8

| Property | Value |
|----------|-------|
| Architecture | YOLOv8n (nano variant) |
| Input Size | 640 × 640 RGB |
| Output | Bounding boxes + confidence |
| mAP | 0.89 on validation set |
| Inference | ~5ms per frame (GPU) |

### Emotion Classification — Custom ResNet-SE CNN

| Property | Value |
|----------|-------|
| Architecture | ResNet with Squeeze-and-Excitation blocks |
| Input Size | 48 × 48 grayscale |
| Classes | 7 (Happy, Sad, Angry, Neutral, Surprise, Fear, Disgust) |
| Parameters | ~4.5M |
| Accuracy | 81% on FER-2013plus test set |
| Key Techniques | Residual connections, SE attention, BatchNorm, Dropout |

### Engagement Score Formula

```
Engagement = Σ (P_emotion × W_emotion) × 100

Weights:
  Happy    = 1.0  (High engagement)
  Surprise = 0.9  (High engagement)
  Neutral  = 0.5  (Passive engagement)
  Angry    = 0.1  (Low engagement)
  Disgust  = 0.1  (Low engagement)
  Fear     = 0.1  (Low engagement)
  Sad      = 0.1  (Low engagement)
```

---

## Training

Training notebooks are available in `training/`:

| Notebook | Description |
|----------|-------------|
| `YOLOv8.ipynb` | Fine-tuning YOLOv8n on custom face detection dataset |
| `classification_training_fer2013.ipynb` | Training the ResNet-SE CNN on FER-2013plus with data augmentation |

### Key Training Details

- **Dataset**: FER-2013plus (corrected labels) — 35,887 images
- **Augmentation**: Random horizontal flip, rotation (±15°), affine transforms
- **Optimizer**: AdamW with cosine annealing LR schedule
- **Regularization**: Dropout (0.5), Label smoothing, SE attention
- **Hardware**: Trained on NVIDIA RTX 3050 (~2 hours)

---

## Performance

| Metric | Value |
|--------|-------|
| End-to-end Latency | ~120ms (GPU) / ~300ms (CPU) |
| Frame Processing Rate | 8 FPS (configurable) |
| YOLOv8 Inference | ~5ms (GPU) / ~50ms (CPU) |
| CNN Inference | ~2ms (GPU) / ~15ms (CPU) |
| WebSocket Overhead | ~10ms (local) / ~100ms (cloud) |
| Frontend Render | 60 FPS (canvas overlay) |
| Detection mAP | 0.89 |
| Classification Accuracy | 81% (FER-2013plus) |

---

## API Reference

### REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check (root) |
| `GET` | `/api/health` | Model status + device info |

### WebSocket Protocol

**Endpoint**: `ws://host:7860/ws/emotion`

**Client → Server** (text message):
```
data:image/jpeg;base64,/9j/4AAQSkZJRg...
```

**Server → Client** (JSON text message):
```json
{
  "emotion": "Happy",
  "score": 78.5,
  "bbox": [120, 80, 280, 320],
  "probabilities": {
    "Angry": 0.02,
    "Disgust": 0.01,
    "Fear": 0.03,
    "Happy": 0.72,
    "Neutral": 0.15,
    "Sad": 0.04,
    "Surprise": 0.03
  }
}
```

---

## Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Ultralytics](https://ultralytics.com/) — YOLOv8 object detection
- [FER-2013plus](https://github.com/microsoft/FERPlus) — Emotion recognition dataset
- [PyTorch](https://pytorch.org/) — Deep learning framework
- [FastAPI](https://fastapi.tiangolo.com/) — Modern Python web framework
- [Next.js](https://nextjs.org/) — React production framework

---

<div align="center">

Built with PyTorch, FastAPI, Next.js, and ❤️

**[Live Demo](https://emotion-analytics.vercel.app)** · **[Report Bug](https://github.com/mugnihidayah/emotion-analytics-realtime/issues)** · **[Request Feature](https://github.com/mugnihidayah/emotion-analytics-realtime/issues)**

</div>