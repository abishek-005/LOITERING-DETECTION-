"""
SENTINEL v2.0 — Flask Application Server
==========================================
Serves the cinematic CCTV HUD frontend, streams processed video frames
via MJPEG, and pushes real-time subject metadata over WebSocket.

Routes:
    GET  /             → index.html (CCTV HUD dashboard)
    GET  /video_feed   → MJPEG stream of processed frames
    WS   /ws           → WebSocket for real-time JSON metadata
    POST /api/upload   → Upload video file
    POST /api/settings → Update threshold / confidence
    POST /api/play     → Resume video processing
    POST /api/pause    → Pause video processing
    GET  /api/stats    → Current session stats (JSON)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `from config import …` works
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    CONFIDENCE_MIN,
    DEVICE,
    LOITERING_TIME_DEFAULT,
    SERVER_HOST,
    SERVER_PORT,
    UPLOAD_FOLDER,
    USE_GPU,
)
from core import LoiteringDetector, PersonDetector, PersonTracker, PoseEstimator
from ui import HUDOverlay

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB upload limit

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared pipeline state (thread-safe via locks)
# ---------------------------------------------------------------------------
_lock = threading.Lock()

# Detection modules — lazily initialised on first video upload
_detector: Optional[PersonDetector] = None
_tracker: Optional[PersonTracker] = None
_pose_estimator: Optional[PoseEstimator] = None
_loitering: Optional[LoiteringDetector] = None
_hud: Optional[HUDOverlay] = None

# Video capture & state
_cap: Optional[cv2.VideoCapture] = None
_is_playing: bool = False
_is_loaded: bool = False
_frame_count: int = 0

# Latest processed frame (shared between processing thread and MJPEG route)
_latest_frame: Optional[np.ndarray] = None
_latest_metadata: dict = {}

# Per-frame subject data for WebSocket consumers
_ws_clients: list = []  # List of active WebSocket connections

# Incident log (in-memory)
_incidents: list = []
_incident_sent: set = set()  # track_ids for which we already sent an incident

# Processing thread reference
_processing_thread: Optional[threading.Thread] = None

# Settings (mutable at runtime)
_settings = {
    "loitering_threshold": LOITERING_TIME_DEFAULT,
    "confidence_threshold": CONFIDENCE_MIN,
}


def _init_pipeline():
    """Lazily initialise detection modules (once)."""
    global _detector, _tracker, _pose_estimator, _loitering, _hud

    if _detector is not None:
        return  # already initialised

    print("[SENTINEL] Initialising detection pipeline…")
    _detector = PersonDetector(confidence=_settings["confidence_threshold"])
    _tracker = PersonTracker()

    # Pose estimation is optional — if MediaPipe is broken the rest of
    # the pipeline (detection → tracking → loitering) must still work.
    try:
        _pose_estimator = PoseEstimator()
        print("[SENTINEL] Pose estimator ready")
    except Exception as exc:
        _pose_estimator = None
        print(f"[SENTINEL] WARNING: Pose estimator failed ({exc}). "
              "Continuing without pose estimation.")

    _loitering = LoiteringDetector(time_threshold=_settings["loitering_threshold"])
    _hud = HUDOverlay()
    print(f"[SENTINEL] Pipeline ready  (device={DEVICE}, gpu={USE_GPU})")


# ═══════════════════════════════════════════════════════════════════════════
# VIDEO PROCESSING THREAD
# ═══════════════════════════════════════════════════════════════════════════

def _processing_loop():
    """Background thread: reads frames, runs the full detection pipeline,
    composites the HUD overlay, and stores results for consumption by
    the MJPEG stream route and WebSocket broadcast.
    """
    global _latest_frame, _latest_metadata, _frame_count, _is_playing, _is_loaded

    fps_counter = 0
    fps_time = time.time()
    current_fps = 0.0

    while True:
        # Wait until we are playing
        if not _is_playing or _cap is None or not _cap.isOpened():
            time.sleep(0.05)
            continue

        ret, frame = _cap.read()
        if not ret:
            # End of video — loop back to start
            _cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            _incident_sent.clear()
            time.sleep(0.05)
            continue

        _frame_count += 1

        # FPS tracking
        fps_counter += 1
        elapsed = time.time() - fps_time
        if elapsed >= 1.0:
            current_fps = fps_counter / elapsed
            fps_counter = 0
            fps_time = time.time()

        # ── 1. Detect ──
        detections = _detector.detect(frame)

        # ── 2. Track ──
        confirmed_tracks = _tracker.update(detections, frame)

        # ── 3. Per-subject processing ──
        tracks_data = []
        subjects_json = []
        active_ids = set()
        alert_count = 0

        for track in confirmed_tracks:
            bbox = tuple(map(int, track.to_ltrb()))
            x1, y1, x2, y2 = bbox
            track_id = str(track.track_id)
            active_ids.add(track_id)

            # Pose estimation (skip if unavailable)
            crop = frame[y1:y2, x1:x2]
            if _pose_estimator is not None:
                landmarks = _pose_estimator.estimate(crop)
                pose_label = _pose_estimator.classify_behavior(landmarks)
            else:
                landmarks = None
                pose_label = "UNKNOWN"

            # Loitering
            is_loitering, duration, threat_level = _loitering.update(track_id, bbox)
            track_duration = _loitering.get_track_duration(track_id)

            if is_loitering:
                alert_count += 1

            # Build track data for HUD
            td = {
                "track_id": track_id,
                "bbox": bbox,
                "is_loitering": is_loitering,
                "landmarks": landmarks,
                "pose_label": pose_label,
            }
            tracks_data.append(td)

            # Build subject JSON for WebSocket
            duration_formatted = _format_duration(duration if is_loitering else track_duration)
            sj = {
                "id": track_id,
                "status": "LOITERING" if is_loitering else "TRACKING",
                "duration": duration if is_loitering else track_duration,
                "duration_formatted": duration_formatted,
                "pose": pose_label,
                "threat": round(threat_level, 2),
                "bbox": list(bbox),
            }
            subjects_json.append(sj)

            # ── Incident logging ──
            if is_loitering and track_id not in _incident_sent:
                _incident_sent.add(track_id)
                incident = {
                    "type": "incident",
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "subject_id": track_id,
                    "event": "LOITERING DETECTED",
                    "duration": duration_formatted,
                    "threat": round(threat_level, 2),
                    "pose": pose_label,
                }
                _incidents.append(incident)
                _broadcast_ws(incident)

        # Cleanup stale IDs
        _loitering.cleanup(active_ids)

        # ── 4. HUD overlay ──
        processed = _hud.process_frame(
            frame, tracks_data, _frame_count, pose_estimator=_pose_estimator
        )

        # ── 5. Store latest frame + metadata ──
        with _lock:
            _latest_frame = processed
            _latest_metadata = {
                "type": "update",
                "subjects": subjects_json,
                "total_subjects": len(subjects_json),
                "active_alerts": alert_count,
                "fps": round(current_fps, 1),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "is_playing": True,
                "frame_number": _frame_count,
            }

        # ── 6. Broadcast metadata via WebSocket ──
        _broadcast_ws(_latest_metadata)

        # Throttle to ~30 FPS max
        time.sleep(0.005)


def _format_duration(seconds: float) -> str:
    """Format seconds as MM:SS."""
    if seconds <= 0:
        return "00:00"
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


# ═══════════════════════════════════════════════════════════════════════════
# WEBSOCKET MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

def _broadcast_ws(data: dict):
    """Send JSON data to all connected WebSocket clients."""
    msg = json.dumps(data)
    dead = []
    for client in _ws_clients:
        try:
            client.send(msg)
        except Exception:
            dead.append(client)
    for c in dead:
        try:
            _ws_clients.remove(c)
        except ValueError:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serve the CCTV HUD dashboard."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG stream of processed video frames."""
    def generate():
        while True:
            with _lock:
                frame = _latest_frame

            if frame is not None:
                _, buffer = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
                )
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buffer.tobytes()
                    + b"\r\n"
                )
            else:
                # No frame yet — send a blank placeholder
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    blank, "AWAITING SIGNAL...", (160, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 65), 2,
                )
                _, buffer = cv2.imencode(".jpg", blank)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buffer.tobytes()
                    + b"\r\n"
                )

            time.sleep(0.033)  # ~30 FPS

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ── WebSocket endpoint (using flask-sock) ──

try:
    from flask_sock import Sock

    sock = Sock(app)

    @sock.route("/ws")
    def ws_endpoint(ws):
        """WebSocket endpoint — keeps connection open and pushes updates."""
        _ws_clients.append(ws)
        print(f"[SENTINEL] WebSocket client connected  (total: {len(_ws_clients)})")

        # Send initial status
        try:
            ws.send(json.dumps({
                "type": "update",
                "subjects": [],
                "total_subjects": 0,
                "active_alerts": 0,
                "fps": 0,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "is_playing": _is_playing,
                "frame_number": 0,
            }))
        except Exception:
            pass

        # Keep alive — read messages (mainly for keeping connection open)
        try:
            while True:
                data = ws.receive(timeout=30)
                if data is None:
                    break
        except Exception:
            pass
        finally:
            if ws in _ws_clients:
                _ws_clients.remove(ws)
            print(f"[SENTINEL] WebSocket client disconnected  (total: {len(_ws_clients)})")

except ImportError:
    print("[SENTINEL] WARNING: flask-sock not installed. WebSocket support disabled.")
    print("[SENTINEL] Install with: pip install flask-sock")


# ── API Endpoints ──

@app.route("/api/upload", methods=["POST"])
def upload_video():
    """Handle video file upload."""
    global _cap, _is_playing, _is_loaded, _frame_count

    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save to upload folder
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    print(f"[SENTINEL] Video uploaded: {filepath}")

    # Initialise pipeline if needed
    _init_pipeline()

    # Open video
    if _cap is not None:
        _cap.release()

    _cap = cv2.VideoCapture(filepath)
    if not _cap.isOpened():
        return jsonify({"error": "Failed to open video file"}), 400

    _frame_count = 0
    _is_loaded = True
    _is_playing = True
    _incident_sent.clear()
    _incidents.clear()

    # Start processing thread if not running
    _ensure_processing_thread()

    return jsonify({
        "status": "ok",
        "filename": file.filename,
        "gpu": USE_GPU,
    })


@app.route("/api/settings", methods=["POST"])
def update_settings():
    """Update detection settings at runtime."""
    data = request.get_json(silent=True) or {}

    if "loitering_threshold" in data:
        threshold = int(data["loitering_threshold"])
        _settings["loitering_threshold"] = threshold
        if _loitering:
            _loitering.set_threshold(threshold)

    if "confidence_threshold" in data:
        confidence = float(data["confidence_threshold"])
        _settings["confidence_threshold"] = confidence
        if _detector:
            _detector.set_confidence(confidence)

    return jsonify({"status": "ok", "settings": _settings})


@app.route("/api/play", methods=["POST"])
def play_video():
    """Resume video processing."""
    global _is_playing
    if _is_loaded:
        _is_playing = True
    return jsonify({"status": "ok", "is_playing": _is_playing})


@app.route("/api/pause", methods=["POST"])
def pause_video():
    """Pause video processing."""
    global _is_playing
    _is_playing = False
    return jsonify({"status": "ok", "is_playing": _is_playing})


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Return current session statistics."""
    with _lock:
        meta = dict(_latest_metadata) if _latest_metadata else {}
    meta["incidents"] = len(_incidents)
    meta["gpu"] = USE_GPU
    meta["device"] = DEVICE
    return jsonify(meta)


@app.route("/api/incidents", methods=["GET"])
def get_incidents():
    """Return all logged incidents."""
    return jsonify({"incidents": _incidents})


# ═══════════════════════════════════════════════════════════════════════════
# THREAD MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

def _ensure_processing_thread():
    """Start the background processing thread if it isn't running."""
    global _processing_thread

    if _processing_thread is not None and _processing_thread.is_alive():
        return

    _processing_thread = threading.Thread(
        target=_processing_loop, daemon=True, name="sentinel-pipeline"
    )
    _processing_thread.start()
    print("[SENTINEL] Processing thread started")


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  SENTINEL v2.0 — AI Surveillance System")
    print(f"  Device: {DEVICE}  |  GPU: {USE_GPU}")
    print(f"  Server: http://localhost:{SERVER_PORT}")
    print("=" * 60)

    app.run(
        host=SERVER_HOST,
        port=SERVER_PORT,
        debug=False,
        threaded=True,
    )
