"""
SENTINEL v2.0 — Central Configuration
======================================
All system-wide constants and settings. Mirrors the original detection
parameters exactly so that swapping modules doesn't change behaviour.
"""

import cv2
import torch
import os

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
USE_GPU: bool = torch.cuda.is_available()
DEVICE: str = 'cuda' if USE_GPU else 'cpu'

# ---------------------------------------------------------------------------
# Detection  (YOLOv8)
# ---------------------------------------------------------------------------
YOLO_MODEL: str = "yolov8n.pt"
CONFIDENCE_MIN: float = 0.5
PERSON_CLASS_ID: int = 0

# ---------------------------------------------------------------------------
# Tracking  (DeepSORT)
# ---------------------------------------------------------------------------
DEEPSORT_MAX_AGE: int = 30
DEEPSORT_N_INIT: int = 3

# ---------------------------------------------------------------------------
# Loitering
# ---------------------------------------------------------------------------
LOITERING_PIXEL_THRESHOLD: int = 30
LOITERING_TIME_DEFAULT: int = 10

# ---------------------------------------------------------------------------
# Pose  (MediaPipe)
# ---------------------------------------------------------------------------
POSE_DETECTION_CONFIDENCE: float = 0.7
POSE_TRACKING_CONFIDENCE: float = 0.7

# ---------------------------------------------------------------------------
# HUD Colors  (BGR for OpenCV)
# ---------------------------------------------------------------------------
HUD_COLORS: dict = {
    "primary":   (0, 255, 65),    # #00FF41  neon green
    "alert":     (0, 165, 255),   # #FFA500  orange
    "critical":  (0, 0, 255),     # #FF0000  red
    "text":      (0, 255, 65),
    "panel_bg":  (10, 10, 10),
    "accent":    (0, 200, 255),
}

# ---------------------------------------------------------------------------
# HUD Rendering
# ---------------------------------------------------------------------------
HUD_FONT: int = cv2.FONT_HERSHEY_SIMPLEX
SCANLINE_ALPHA: float = 0.05
PANEL_ALPHA: float = 0.6

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
SERVER_HOST: str = '0.0.0.0'
SERVER_PORT: int = 5000
UPLOAD_FOLDER: str = os.path.join(os.path.dirname(__file__), 'uploads')
