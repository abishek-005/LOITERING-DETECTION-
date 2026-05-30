"""
SENTINEL v2.0 — Cinematic HUD Overlay
======================================
OpenCV drawing operations applied to video frames *before* they are
streamed to the browser.  Every method uses alpha-blended overlays so
that the base image shows through semi-transparent UI chrome.

The visual signature is a **green-tinted grayscale** feed (night-vision
CCTV aesthetic) with tactical crosshair bounding boxes, animated scan
lines, and top/bottom information bars.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np

from config import HUD_COLORS, HUD_FONT, SCANLINE_ALPHA, PANEL_ALPHA, DEVICE


class HUDOverlay:
    """Stateless HUD renderer — call :meth:`process_frame` each tick."""

    # Cache the vignette mask so we only compute it once per resolution
    _vignette_cache: Dict[tuple, np.ndarray] = {}

    # -----------------------------------------------------------------
    # Colour grading
    # -----------------------------------------------------------------
    @staticmethod
    def apply_grayscale_green_tint(frame: np.ndarray) -> np.ndarray:
        """Convert *frame* to grayscale and re-map into a green-channel
        dominated BGR image, producing the classic night-vision look.

        The green channel carries 100 % of the luminance, blue and red
        receive ~30 % and ~15 % respectively so the image is clearly
        tinted but not *purely* monochrome green.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Map luminance into BGR channels with a green bias
        b = (gray * 0.30).astype(np.uint8)
        g = gray  # full luminance in green
        r = (gray * 0.15).astype(np.uint8)
        return cv2.merge([b, g, r])

    # -----------------------------------------------------------------
    # Tactical bounding boxes
    # -----------------------------------------------------------------
    @staticmethod
    def draw_crosshair_box(
        frame: np.ndarray,
        bbox: tuple,
        color: tuple,
        thickness: int = 2,
    ) -> None:
        """Draw L-shaped corner brackets instead of a full rectangle.

        Each bracket arm is ≈ 20 % of the box width / height, giving a
        clean tactical-HUD look.
        """
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        arm_w = max(int(w * 0.20), 6)
        arm_h = max(int(h * 0.20), 6)

        # Top-left
        cv2.line(frame, (x1, y1), (x1 + arm_w, y1), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + arm_h), color, thickness, cv2.LINE_AA)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - arm_w, y1), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y1 + arm_h), color, thickness, cv2.LINE_AA)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + arm_w, y2), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - arm_h), color, thickness, cv2.LINE_AA)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - arm_w, y2), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - arm_h), color, thickness, cv2.LINE_AA)

    # -----------------------------------------------------------------
    # Subject label
    # -----------------------------------------------------------------
    @staticmethod
    def draw_subject_label(
        frame: np.ndarray,
        bbox: tuple,
        track_id: str,
        is_loitering: bool,
        pose_label: str = "",
    ) -> None:
        """Draw the track-ID label (and optional pose tag) above the box.

        Green for normal subjects, orange for loitering alerts.
        """
        x1, y1, _, _ = bbox
        color = HUD_COLORS["alert"] if is_loitering else HUD_COLORS["primary"]

        label = f"ID:{track_id}"
        if pose_label:
            label += f"  [{pose_label}]"
        if is_loitering:
            label += "  LOITERING"

        # Background pill for readability
        (tw, th), _ = cv2.getTextSize(label, HUD_FONT, 0.45, 1)
        pad = 4
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x1, y1 - th - 2 * pad),
            (x1 + tw + 2 * pad, y1),
            HUD_COLORS["panel_bg"],
            -1,
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(
            frame,
            label,
            (x1 + pad, y1 - pad),
            HUD_FONT,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    # -----------------------------------------------------------------
    # Animated scan line
    # -----------------------------------------------------------------
    @staticmethod
    def draw_scan_line(frame: np.ndarray, frame_count: int) -> None:
        """Draw a semi-transparent horizontal green line that sweeps
        from top to bottom, cycling every ~300 frames.
        """
        h, w = frame.shape[:2]
        y = frame_count % h  # simple modulo sweep

        overlay = frame.copy()
        cv2.line(overlay, (0, y), (w, y), HUD_COLORS["primary"], 1, cv2.LINE_AA)
        # A second, fainter trailing line
        y2 = (y - 4) % h
        cv2.line(overlay, (0, y2), (w, y2), HUD_COLORS["primary"], 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, SCANLINE_ALPHA + 0.15, frame, 1 - SCANLINE_ALPHA - 0.15, 0, frame)

    # -----------------------------------------------------------------
    # Top chrome bar
    # -----------------------------------------------------------------
    @staticmethod
    def draw_top_chrome(frame: np.ndarray) -> None:
        """Render the top status bar:

            ◉ LIVE   |   2026-05-29 14:58:00   |   REC ●
        """
        h, w = frame.shape[:2]
        bar_h = 32
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), HUD_COLORS["panel_bg"], -1)
        cv2.addWeighted(overlay, PANEL_ALPHA, frame, 1 - PANEL_ALPHA, 0, frame)

        now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        text = f"\u25C9 LIVE   |   {now}   |   REC \u25CF"
        cv2.putText(
            frame, text, (12, 22), HUD_FONT, 0.45,
            HUD_COLORS["primary"], 1, cv2.LINE_AA,
        )

    # -----------------------------------------------------------------
    # Bottom chrome bar
    # -----------------------------------------------------------------
    @staticmethod
    def draw_bottom_chrome(
        frame: np.ndarray,
        subject_count: int,
        alert_count: int,
    ) -> None:
        """Render the bottom status bar:

            SENTINEL v2.0  |  SUBJECTS: n  |  ALERTS: n  |  device
        """
        h, w = frame.shape[:2]
        bar_h = 32
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), HUD_COLORS["panel_bg"], -1)
        cv2.addWeighted(overlay, PANEL_ALPHA, frame, 1 - PANEL_ALPHA, 0, frame)

        device_label = DEVICE.upper()
        text = (
            f"SENTINEL v2.0  |  SUBJECTS: {subject_count}  "
            f"|  ALERTS: {alert_count}  |  {device_label}"
        )
        cv2.putText(
            frame, text, (12, h - 10), HUD_FONT, 0.45,
            HUD_COLORS["text"], 1, cv2.LINE_AA,
        )

    # -----------------------------------------------------------------
    # Vignette
    # -----------------------------------------------------------------
    @classmethod
    def apply_vignette(cls, frame: np.ndarray) -> np.ndarray:
        """Darken the frame corners for a cinematic vignette effect.

        The mask is computed once per resolution and cached.
        """
        h, w = frame.shape[:2]
        key = (w, h)

        if key not in cls._vignette_cache:
            # Build a radial gradient mask (1.0 at centre, ~0.3 at corners)
            x = np.linspace(-1, 1, w, dtype=np.float32)
            y = np.linspace(-1, 1, h, dtype=np.float32)
            xv, yv = np.meshgrid(x, y)
            radius = np.sqrt(xv ** 2 + yv ** 2)
            # Normalise: centre = 1.0, max corner ≈ sqrt(2)
            mask = 1.0 - np.clip(radius / np.sqrt(2), 0, 1) * 0.7
            # Expand to 3-channel
            cls._vignette_cache[key] = np.dstack([mask, mask, mask])

        vignette_mask = cls._vignette_cache[key]
        return (frame.astype(np.float32) * vignette_mask).astype(np.uint8)

    # -----------------------------------------------------------------
    # Full pipeline
    # -----------------------------------------------------------------
    def process_frame(
        self,
        frame: np.ndarray,
        tracks_data: List[dict],
        frame_count: int,
        pose_estimator=None,
    ) -> np.ndarray:
        """Apply the complete HUD pipeline and return the composited frame.

        Parameters
        ----------
        frame : numpy.ndarray
            Raw BGR video frame.
        tracks_data : list[dict]
            Each dict must contain:

            * ``track_id``     – str
            * ``bbox``         – (x1, y1, x2, y2) ints
            * ``is_loitering`` – bool
            * ``landmarks``    – MediaPipe landmarks or ``None``
            * ``pose_label``   – str (e.g. ``'STANDING'``)
        frame_count : int
            Monotonically increasing counter used to animate the scan
            line.
        pose_estimator : PoseEstimator, optional
            If provided, skeletons are drawn via its ``draw_skeleton``
            method.

        Returns
        -------
        numpy.ndarray
            The final composited BGR frame, ready for encoding /
            streaming.
        """
        # 1. Colour grading — night-vision green tint
        out = self.apply_grayscale_green_tint(frame)

        # 2. Per-subject overlays
        alert_count = 0
        for td in tracks_data:
            bbox = td["bbox"]
            is_loitering = td.get("is_loitering", False)
            track_id = td.get("track_id", "?")
            landmarks = td.get("landmarks")
            pose_label = td.get("pose_label", "")

            if is_loitering:
                alert_count += 1

            # Crosshair box colour
            box_color = (
                HUD_COLORS["alert"] if is_loitering else HUD_COLORS["primary"]
            )
            self.draw_crosshair_box(out, bbox, box_color, thickness=2)

            # Subject label
            self.draw_subject_label(out, bbox, track_id, is_loitering, pose_label)

            # Pose skeleton
            if pose_estimator and landmarks:
                skel_color = (
                    HUD_COLORS["alert"] if is_loitering else HUD_COLORS["accent"]
                )
                pose_estimator.draw_skeleton(out, bbox, landmarks, color=skel_color)

        # 3. Scan line
        self.draw_scan_line(out, frame_count)

        # 4. Chrome bars
        self.draw_top_chrome(out)
        self.draw_bottom_chrome(out, len(tracks_data), alert_count)

        # 5. Vignette
        out = self.apply_vignette(out)

        return out
