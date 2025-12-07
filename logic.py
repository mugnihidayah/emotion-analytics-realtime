# logic.py
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO # type: ignore
from collections import deque
from torchvision import transforms
from PIL import Image

# CNN ARCHITECTURE

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualSEBlock(nn.Module):
    """Residual Block integrated with SE Block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.se = SEBlock(out_channels)
        self.shortcut_path = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut_path = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
    
    def forward(self, x):
        out = self.main_path(x)
        out = self.se(out)
        shortcut = self.shortcut_path(x)
        return F.relu(out + shortcut)


class cnn_model(nn.Module):
    def __init__(self, dropout_rate=0.5, num_classes=7):
        super().__init__()
        self.in_channels = 64

        # initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(1, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # ResNet Stages
        self.stage1 = self._make_stage(64, num_blocks=3, stride=1)
        self.stage2 = self._make_stage(128, num_blocks=3, stride=2)
        self.stage3 = self._make_stage(256, num_blocks=3, stride=2)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes),
        )

    def _make_stage(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(ResidualSEBlock(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.classifier(x)
        return x

# EMOTION ANALYZER

class EmotionAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Logic loaded on: {self.device}")

        # Load Models
        self.yolo = YOLO("./models/best.pt")
        
        # Load CNN
        self.emotion_model = cnn_model(num_classes=7).to(self.device)
        try:
            weights = torch.load("./models/model_cnn.pth", map_location=self.device)
            self.emotion_model.load_state_dict(weights)
        except FileNotFoundError:
            raise FileNotFoundError("CRITICAL: ./models/model_cnn.pth not found!")
            
        self.emotion_model.eval()

        self.labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
        
        # History for probability smoothing
        self.prob_history = deque(maxlen=5)

        self.transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((48,48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,))
        ])

        # Engagement Weight
        self.engagement_weights = np.array([
            0.1,  # Angry
            0.1,  # Disgust
            0.1,  # Fear
            1.0,  # Happy (High Engagement)
            0.5,  # Neutral (Passive Engagement)
            0.1,  # Sad
            0.9   # Surprise (High Engagement)
        ])

    def process_frame(self, frame):
        small = cv2.resize(frame, (640,480))
        results = self.yolo(small, verbose=False, conf=0.5)

        if len(results[0].boxes) == 0:
            return small, "No Face", 0.0, None

        boxes = results[0].boxes.xyxy.cpu().numpy()
        areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
        idx = np.argmax(areas)
        x1,y1,x2,y2 = boxes[idx].astype(int)

        # Padding & Boundary Check
        h,w,_ = small.shape
        x1,y1 = max(0,x1), max(0,y1)
        x2,y2 = min(w,x2), min(h,y2)

        roi = small[y1:y2, x1:x2]
        if roi.size == 0:
            return small, "No Face", 0.0, None

        # Preprocessing CNN
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        pil = Image.fromarray(img)
        tensor = self.transform(pil).unsqueeze(0).to(self.device) # type: ignore

        # Inference
        with torch.no_grad():
            out = self.emotion_model(tensor)
            probs = F.softmax(out, dim=1).cpu().numpy()[0]

        # Temporal Smoothing
        self.prob_history.append(probs)
        avg_probs = np.mean(self.prob_history, axis=0)

        # Determine the Dominant Label
        emo_id = np.argmax(avg_probs)
        emotion = self.labels[emo_id]

        # Calculate Real Engagement Score (0 - 100)
        raw_score = np.sum(avg_probs * self.engagement_weights)
        final_score = np.clip(raw_score * 100, 0, 100)

        return small, emotion, final_score, (x1,y1,x2,y2)