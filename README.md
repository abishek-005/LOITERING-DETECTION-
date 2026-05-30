# 👁️ Loitering & Pose Detection — Surveillance AI

An advanced, real-time smart surveillance system built using **Streamlit**. This application integrates **YOLOv8** for precise object detection, **Deep SORT** for multi-object tracking, and **MediaPipe** for full-body pose estimation to autonomously flag loitering behavior.

---

## ✨ Features

* 🎯 **Precision Detection:** Real-time human target identification via YOLOv8.
* 🔁 **Multi-Object Tracking:** Seamless identity retention across frames using Deep SORT.
* 🧭 **Pose Estimation:** Extracts and visualizes frame-by-frame body landmarks using MediaPipe.
* ⏱️ **Smart Loitering Analytics:** Tracks individual bounding-box center drift. If a unique `track_id` remains within a designated pixel radius beyond a custom duration threshold, a **LOITERING ALERT** is flagged instantly.
* 🖥️ **Interactive Sidebar Control:** Tweak detection confidence, change time thresholds dynamically, and drop your own `.mp4`/`.avi` files right through the Streamlit UI.

---

## 📊 Demo Preview

<img width="1339" height="618" alt="image" src="https://github.com/user-attachments/assets/9ac8e4a6-b67a-425d-b217-be870aa4cb1f" />

<img width="1325" height="446" alt="image" src="https://github.com/user-attachments/assets/fe331b04-d30f-47ce-8a2e-961c78228e00" />

<img width="1315" height="194" alt="image" src="https://github.com/user-attachments/assets/abcf73e8-7c3c-4497-8be3-01e7ce944cdc" />

<img width="771" height="308" alt="image" src="https://github.com/user-attachments/assets/6c0ba643-7d40-4906-8821-3af23516a92d" />



---

## 🚀 Quick Start

### 1. Repository Setup
```bash
git clone [https://github.com/abishek-005/loitering-pose-detection.git](https://github.com/abishek-005/loitering-pose-detection.git)
cd loitering-pose-detection
```
### 2. Environment InitializationBash# Initialize and activate virtual environment
```bash
python -m venv venv
```
### Windows:
```
venv\Scripts\activate
```
### 3. Setup Requirements & WeightsDownload standard weights (yolov8n.pt) from Ultralytics and drop them right in the project root directory.Install standard dependencies:
```bash
install -r requirements.txt
```

### 4. Boot Up the DashboardBashstreamlit run app.py
* Open http://localhost:8501 on your web browser to test your video streams

* 🛠️ Configuration Settings (Sidebar UI)Configuration ParameterOperational Range / Input TypeFunctional MechanismLoitering ThresholdSeconds (Default: 10s)Determines the minimum continuous static duration required to trigger alerts.Detection ConfidenceSlider (0.3 – 1.0)YOLOv8 operational confidence filtering threshold for human detection.

* Video Upload SourceFile Drag & Drop (.mp4, .avi, .mov)Dynamically pipes local footage variables into the detection architecture.🧠 How the Core Detection Logic WorksSpatial Tracking: The engine assigns a persistent, locked track_id to each individual person detected in the frame.Drift Computation: The application monitors the absolute centroid vectors of the tracking bounding box relative to its initial timestamp coordinates.Alert Trigger: If spatial coordinates stay locked within a 30px spatial radius longer than the configured time threshold, the engine highlights the target stream with a LOITERING ALERT rectangle.

* Any macro movement resets the evaluation clock.🔍 Troubleshooting NotesReduced Framerates (Low FPS): By default, processing falls back to the CPU. For hardware acceleration, ensure a CUDA-enabled PyTorch environment is configured (torch + torchvision compiled with matching CUDA toolkit).DeepSORT Embedder Failures: If using a device without a dedicated GPU, initialize the tracker module using the explicit flag: DeepSort(..., embedder_gpu=False).MediaPipe Node Errors: Double-check your environment binary wheels if deploying inside distinct container models or virtual setups.

## 👨‍💻 Author

abishek-005

AI/ML Developer | Python Developer

📫 Contact: ak.abishek005@gmail.com🌐 GitHub: @abishek-005
