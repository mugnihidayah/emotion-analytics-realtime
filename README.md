# Real-time Emotion Analytics Dashboard

A deep learning-powered system for real-time facial emotion recognition and engagement analytics using CUDA-accelerated computer vision.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-ee4c2c?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF)

## Overview

This project goes beyond basic emotion classification by providing real-time engagement analytics with smooth, production-ready visualizations. The system uses a dual-model pipeline combining YOLOv8 for face detection and a custom CNN for emotion recognition, all optimized for CUDA acceleration.

## Features

- **CUDA-Accelerated Inference**: Parallel processing pipeline that maintains 30+ FPS on modern GPUs
- **Engagement Analytics**: Dynamic scoring system based on weighted emotion probabilities
- **Signal Smoothing**: Exponential Moving Average (EMA) filters to eliminate prediction jitter
- **Live Visualization**: OpenCV-rendered HUD with real-time sparkline graphs
- **Modern UI**: Custom Streamlit dashboard with full-width video feed

## How It Works

The system processes video frames through two neural networks:

1. **YOLOv8** detects faces and extracts regions of interest
2. **Custom ResNet-SE CNN** classifies emotions from detected faces
3. **Engagement Calculator** computes weighted scores from softmax probabilities
4. **EMA Filter** smooths predictions and bounding boxes
5. **OpenCV Renderer** overlays analytics directly on video frames

**Engagement Score Formula:**

```
Score = Σ (P_emotion × W_emotion)
```

Where `P` is the prediction probability and `W` is the engagement weight (e.g., Happy=1.0, Neutral=0.5, Sad=0.1).

## Installation

### Prerequisites

- Python 3.9 or higher
- NVIDIA GPU with CUDA support (recommended)
- Latest NVIDIA drivers

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/emotion-analytics-realtime.git
cd emotion-analytics-realtime
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install PyTorch with CUDA support:

Check your CUDA version with `nvidia-smi`, then install the appropriate version:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

4. Install remaining dependencies:
```bash
pip install -r requirements.txt
```

5. Download model weights:

Place the following files in the `models/` directory:
- `best.pt` - YOLOv8 face detection model
- `model_cnn.pth` - Emotion classification model

## Usage

Start the Streamlit application:

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`. Grant camera permissions when prompted, and the real-time analysis will begin automatically.

## Project Structure

```
emotion-analytics-realtime/
│
├── models/
│   ├── best.pt                    # YOLOv8 face detector
│   └── model_cnn.pth              # Emotion classifier
│
├── training/
│   ├── train_emotion_cnn.ipynb    # CNN training notebook
│   └── train_yolo_face.ipynb      # YOLOv8 training notebook
│
├── app.py                          # Streamlit UI
├── logic.py                        # Inference engine
├── requirements.txt                # Python dependencies
└── README.md
```

## Model Details

### Face Detection
- **Architecture**: YOLOv8n (nano variant)
- **Input**: 640×640 RGB images
- **Output**: Bounding boxes with confidence scores

### Emotion Classification
- **Architecture**: Custom ResNet with SE-Blocks
- **Input**: 48×48 grayscale face crops
- **Classes**: Happy, Sad, Angry, Neutral, Surprise, Fear, Disgust
- **Techniques**: Residual connections, Squeeze-and-Excitation, Batch Normalization

## Training

Training notebooks are available in the `training/` folder:

- `YOLOv8.ipynb`: Fine-tuning YOLOv8 on custom face datasets
- `classification_training_fer2013.ipynb`: Training the emotion classifier with data augmentation and SE-blocks

## Performance

- **Inference Speed**: 30-60 FPS on RTX 3050 (depending on resolution)
- **Detection mAP**: 0.89 on validation set
- **Classification Accuracy**: 81% on FER-2013plus test set

## Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 by Ultralytics
- FER-2013plus dataset for emotion recognition
- Streamlit team for the amazing framework

---

Built with PyTorch, OpenCV, and Streamlit