import cv2
import torch
import math
import time
import tempfile
import streamlit as st
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import mediapipe as mp

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Loitering Detection App", layout="wide")
st.title("🚨 Loitering & Pose Detection - Surveillance AI")

st.sidebar.header("⚙️ Settings")
threshold_sec = st.sidebar.slider("Loitering Threshold (seconds)", 5, 60, 10)
confidence_threshold = st.sidebar.slider("Detection Confidence", 0.3, 1.0, 0.5)
video_file = st.sidebar.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

# -------------------- Models Setup --------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.sidebar.success(f"Using device: {device}")

model = YOLO("yolov8n.pt").to(device)
tracker = DeepSort(max_age=30, n_init=3, embedder="mobilenet", embedder_gpu=True)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

loitering_tracker = {}

def detect_loitering(track_id, bbox, threshold_sec=10):
    x1, y1, x2, y2 = bbox
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    now = time.time()

    if track_id not in loitering_tracker:
        loitering_tracker[track_id] = {'pos': center, 'start': now}
        return False

    prev = loitering_tracker[track_id]
    dist = math.hypot(center[0] - prev['pos'][0], center[1] - prev['pos'][1])

    if dist < 30:
        if now - prev['start'] > threshold_sec:
            return True
    else:
        loitering_tracker[track_id] = {'pos': center, 'start': now}

    return False


# -------------------- Video Player --------------------
if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()
    fps_info = st.sidebar.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]
        detections = []

        for box, cls, score in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            if int(cls) == 0 and score.item() > confidence_threshold:
                x1, y1, x2, y2 = map(int, box.tolist())
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], score.item(), 'person'))

        tracks = tracker.update_tracks(detections, frame=frame)
        track_landmarks = {}

        for track in tracks:
            if not track.is_confirmed():
                continue

            x1, y1, x2, y2 = map(int, track.to_ltrb())
            track_id = track.track_id

            # Loitering Detection
            if detect_loitering(track_id, (x1, y1, x2, y2), threshold_sec):
                cv2.putText(frame, "LOITERING!", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Pose Detection
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)
                track_landmarks[track_id] = result.pose_landmarks

        # Convert to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()

else:
    st.info("👆 Please upload a video file to start detection.")
