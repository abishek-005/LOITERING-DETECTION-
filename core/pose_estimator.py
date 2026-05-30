"""
SENTINEL v2.0 — Pose Estimation (MediaPipe)
============================================
Wraps ``mediapipe.solutions.pose`` with the same confidence settings
used in the original script:

    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

The core ``estimate()`` method reproduces the original crop→RGB→process
pipeline.  ``classify_behavior()`` and ``draw_skeleton()`` are new
additions for the tactical HUD.

If MediaPipe is unavailable (e.g. broken dependency chain), the class
degrades gracefully — returning ``None`` landmarks and ``'UNKNOWN'``
pose labels — so that the rest of the pipeline (detection, tracking,
loitering) continues to function.
"""

from __future__ import annotations

from typing import Optional

import cv2

from config import (
    POSE_DETECTION_CONFIDENCE,
    POSE_TRACKING_CONFIDENCE,
    HUD_COLORS,
)

# ---------------------------------------------------------------------------
# Graceful MediaPipe import — the rest of SENTINEL must not crash if
# mediapipe or its transitive dependency (tensorflow / numpy) is broken.
# ---------------------------------------------------------------------------
_MEDIAPIPE_AVAILABLE = False
_mp = None

try:
    import mediapipe as _mp
    # Verify that the legacy `solutions` API is present (removed in newer
    # mediapipe wheels that ship only the `tasks` API).
    if hasattr(_mp, "solutions") and hasattr(_mp.solutions, "pose"):
        _MEDIAPIPE_AVAILABLE = True
    else:
        print("[SENTINEL] WARNING: mediapipe installed but 'solutions.pose' "
              "not found. Pose estimation disabled.")
except Exception as exc:
    print(f"[SENTINEL] WARNING: Failed to import mediapipe ({exc}). "
          "Pose estimation disabled — detection & loitering still active.")


class PoseEstimator:
    """MediaPipe Pose wrapper with behaviour classification.

    If MediaPipe is unavailable the estimator is still instantiable but
    every method returns a safe default (``None`` / ``'UNKNOWN'``).
    """

    def __init__(self) -> None:
        self._available = _MEDIAPIPE_AVAILABLE
        if self._available:
            self.mp_pose = _mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=POSE_DETECTION_CONFIDENCE,
                min_tracking_confidence=POSE_TRACKING_CONFIDENCE,
            )
            self.mp_drawing = _mp.solutions.drawing_utils
        else:
            self.mp_pose = None
            self.pose = None
            self.mp_drawing = None

    # ------------------------------------------------------------------
    # Core estimation — identical to original lines 97-101
    # ------------------------------------------------------------------
    def estimate(self, frame_crop):
        """Run pose estimation on a cropped frame region.

        Parameters
        ----------
        frame_crop : numpy.ndarray
            BGR image crop (person bounding-box region).

        Returns
        -------
        pose_landmarks or None
            MediaPipe ``NormalizedLandmarkList`` if a pose is found.
        """
        if not self._available:
            return None
        if frame_crop.size == 0:
            return None
        rgb = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        return result.pose_landmarks

    # ------------------------------------------------------------------
    # Behaviour classification (new for SENTINEL v2.0)
    # ------------------------------------------------------------------
    def classify_behavior(self, landmarks) -> str:
        """Classify pose into a human-readable behaviour label.

        Parameters
        ----------
        landmarks
            ``NormalizedLandmarkList`` returned by :meth:`estimate`.

        Returns
        -------
        str
            One of ``'STANDING'``, ``'WALKING'``, ``'CROUCHING'``,
            ``'SITTING'``, or ``'UNKNOWN'``.
        """
        if not self._available or landmarks is None:
            return "UNKNOWN"

        try:
            lm = landmarks.landmark

            # Key anatomical points
            left_hip = lm[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = lm[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            left_knee = lm[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
            right_knee = lm[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
            left_shoulder = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_ankle = lm[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
            right_ankle = lm[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

            # Average Y positions (normalised 0-1, top → bottom)
            hip_y = (left_hip.y + right_hip.y) / 2
            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            knee_y = (left_knee.y + right_knee.y) / 2
            ankle_y = (left_ankle.y + right_ankle.y) / 2

            # Segment lengths in normalised coords
            torso_length = abs(hip_y - shoulder_y)
            leg_length = abs(ankle_y - hip_y)

            # Hip-knee ratio for crouch detection
            hip_knee_ratio = abs(knee_y - hip_y) / max(torso_length, 0.01)

            if torso_length > 0 and leg_length > 0:
                ratio = torso_length / leg_length
                if ratio > 1.2:
                    # Torso relatively long vs legs → crouching or sitting
                    if hip_knee_ratio < 0.3:
                        return "SITTING"
                    return "CROUCHING"
                elif hip_knee_ratio < 0.5:
                    return "CROUCHING"
                else:
                    # Check ankle spread to distinguish walking vs standing
                    ankle_spread = abs(left_ankle.x - right_ankle.x)
                    if ankle_spread > 0.15:
                        return "WALKING"
                    return "STANDING"

            return "STANDING"
        except (IndexError, AttributeError):
            return "UNKNOWN"

    # ------------------------------------------------------------------
    # Skeleton drawing for HUD
    # ------------------------------------------------------------------
    def draw_skeleton(
        self,
        frame,
        bbox: tuple,
        landmarks,
        color: Optional[tuple] = None,
    ) -> None:
        """Draw pose skeleton on *frame* mapped into the bounding box.

        Parameters
        ----------
        frame : numpy.ndarray
            Full video frame (BGR).
        bbox : tuple
            ``(x1, y1, x2, y2)`` pixel coordinates of the person box.
        landmarks
            ``NormalizedLandmarkList`` from :meth:`estimate`.
        color : tuple, optional
            BGR colour; defaults to ``HUD_COLORS['primary']``.
        """
        if not self._available or landmarks is None:
            return

        x1, y1, x2, y2 = bbox
        c = color or HUD_COLORS["primary"]

        h, w = y2 - y1, x2 - x1
        connections = self.mp_pose.POSE_CONNECTIONS

        # Draw connections
        for connection in connections:
            start_lm = landmarks.landmark[connection[0]]
            end_lm = landmarks.landmark[connection[1]]

            start_point = (int(start_lm.x * w + x1), int(start_lm.y * h + y1))
            end_point = (int(end_lm.x * w + x1), int(end_lm.y * h + y1))

            cv2.line(frame, start_point, end_point, c, 1, cv2.LINE_AA)

        # Draw keypoints
        for lm_point in landmarks.landmark:
            px = int(lm_point.x * w + x1)
            py = int(lm_point.y * h + y1)
            cv2.circle(frame, (px, py), 2, c, -1, cv2.LINE_AA)
