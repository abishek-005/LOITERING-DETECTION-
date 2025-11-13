Loitering & Pose Detection — Surveillance AI


A Streamlit app that combines YOLOv8 person detection, Deep SORT tracking and MediaPipe pose estimation to detect loitering (people staying in roughly the same position for a configurable time) and show real-time pose landmarks. Ready for local use and easy to push to GitHub.


Features

🎯 Real-time person detection using YOLOv8


🔁 Multi-object tracking with Deep SORT


🧭 Pose estimation via MediaPipe


⏱️ Loitering detection with configurable time threshold


🖥️ Streamlit UI for quick configuration and visualization


Demo
(Place a short GIF or screenshot here showing detection + loitering alert)

Quick start

Clone the repo:

git clone https://github.com/abishek-005/loitering-pose-detection.git
cd loitering-pose-detection


Create a virtual environment and install dependencies:

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate


Put yolov8n.pt in the repo root (or edit the script to point to your model path). You can download the small YOLOv8 weights from Ultralytics if you don't already have them.

Run the app:

streamlit run app.py
# or whatever filename you used, e.g.
streamlit run loitering_streamlit.py


Open http://localhost:8501 (Streamlit will show the URL).



Requirements

Minimum recommended packages (put these in requirements.txt):

ultralytics
torch
opencv-python
streamlit
deep_sort_realtime
mediapipe
numpy


Tip: Install a CUDA-enabled PyTorch build if you want GPU acceleration (torch + matching CUDA). If no GPU is available the code will fall back to CPU but slower.

Example requirements.txt:

ultralytics>=8.0
torch>=1.13
opencv-python
streamlit
deep_sort_realtime
mediapipe
numpy

Configuration (Streamlit sidebar)

Loitering Threshold (seconds) — how long a tracked person must stay roughly in the same place to be labelled loitering (default 10s).

Detection Confidence — YOLO confidence cutoff for person detection (0.3–1.0).

Upload Video — upload .mp4, .avi, .mov to analyze.

How loitering detection works (brief)

The tracker assigns a unique track_id to each person.

For each track, we compute the bounding-box center and record the first timestamp.

If the person’s center doesn't move beyond a small pixel threshold (30 px in the code) for longer than the configured threshold (e.g., 10s), we flag them as LOITERING and draw an alert rectangle.

Movement above the threshold resets the timer for that track.

You can tune the pixel distance (30) and threshold_sec to suit camera resolution, field of view and required sensitivity.

Pose detection notes

Pose estimation is applied on the cropped person bounding box using MediaPipe Pose.

result.pose_landmarks is stored per track and can be used to implement activity classification (sitting, standing, falling, etc.) later.

Cropping small bounding boxes may produce noisy/no-detections — consider filtering small boxes.

Performance tips & caveats

GPU recommended for YOLO + DeepSort embedder to get smooth processing.

deep_sort_realtime embedder options: if you do not have GPU, set embedder_gpu=False when initializing tracker to avoid GPU-only errors.

For long videos use a smaller YOLO model (e.g., yolov8n) or reduce input resolution to improve FPS.

Lighting, camera angle and crowd density affect detection/tracking quality — you may need to tune thresholds.

Troubleshooting

No detections / very slow: ensure yolov8n.pt is accessible and Torch installed correctly. If on CPU, expect reduced FPS.

DeepSort embedder errors: try DeepSort(..., embedder_gpu=False) if you don’t have a CUDA-capable GPU or appropriate CUDA Torch build.

MediaPipe errors: ensure mediapipe installed; some systems require extra packages—use pip binary wheels.

Extend / Ideas

Add real-time alerts (email / Webhook / MQTT) when loitering is detected.

Save loitering events to a CSV / database (track_id, bbox, start_time, end_time).

Add a dashboard with statistics (total loitering per hour, heatmap).

Use pose landmarks to detect suspicious activities or falls.
