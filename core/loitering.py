"""
SENTINEL v2.0 — Loitering Detection
=====================================
Encapsulates the original ``detect_loitering()`` function as a stateful
class.  The core decision logic is *identical*:

    center = ((x1+x2)//2, (y1+y2)//2)
    dist   = math.hypot(…)
    if dist < 30:          # pixel threshold
        if elapsed > threshold_sec:
            → loitering

The class adds ``duration``, ``threat_level``, lifetime tracking, and
stale-ID cleanup — all used by the HUD but **not** altering the
detection decision.
"""

from __future__ import annotations

import math
import time
from typing import Dict, Set, Tuple

from config import LOITERING_PIXEL_THRESHOLD, LOITERING_TIME_DEFAULT


class LoiteringDetector:
    """Stateful loitering detector — one instance per pipeline.

    Parameters
    ----------
    pixel_threshold : int, optional
        Movement distance (px) below which a subject is considered
        stationary (default ``config.LOITERING_PIXEL_THRESHOLD``).
    time_threshold : int | float, optional
        Seconds of stationary presence before a loitering alert fires
        (default ``config.LOITERING_TIME_DEFAULT``).
    """

    def __init__(
        self,
        pixel_threshold: int | None = None,
        time_threshold: int | float | None = None,
    ) -> None:
        self.pixel_threshold: int = pixel_threshold or LOITERING_PIXEL_THRESHOLD
        self.time_threshold: float = float(time_threshold or LOITERING_TIME_DEFAULT)
        self._tracker: Dict[str, dict] = {}  # track_id → {pos, start, first_seen}

    # ------------------------------------------------------------------
    # Core update — mirrors original detect_loitering() lines 32-50
    # ------------------------------------------------------------------
    def update(
        self, track_id: str, bbox: tuple
    ) -> Tuple[bool, float, float]:
        """Update tracking state for a single subject.

        Parameters
        ----------
        track_id : str
            Unique track identifier from DeepSORT.
        bbox : tuple
            ``(x1, y1, x2, y2)`` pixel bounding box.

        Returns
        -------
        tuple[bool, float, float]
            * **is_loitering** — ``True`` when the subject has been
              stationary longer than ``time_threshold``.
            * **duration** — seconds since the subject became stationary
              (or 0.0 if they just moved).
            * **threat_level** — 0.0 → 1.0 (scales linearly over 3×
              the loitering threshold).
        """
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        now = time.time()

        if track_id not in self._tracker:
            self._tracker[track_id] = {
                "pos": center,
                "start": now,
                "first_seen": now,
            }
            return False, 0.0, 0.0

        prev = self._tracker[track_id]
        dist = math.hypot(
            center[0] - prev["pos"][0],
            center[1] - prev["pos"][1],
        )

        if dist < self.pixel_threshold:
            duration = now - prev["start"]
            is_loitering = duration > self.time_threshold
            threat_level = (
                min(1.0, duration / (self.time_threshold * 3))
                if is_loitering
                else 0.0
            )
            return is_loitering, duration, threat_level
        else:
            self._tracker[track_id] = {
                "pos": center,
                "start": now,
                "first_seen": prev.get("first_seen", now),
            }
            return False, 0.0, 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_track_duration(self, track_id: str) -> float:
        """Total time (seconds) a subject has been under observation."""
        if track_id in self._tracker:
            return time.time() - self._tracker[track_id].get(
                "first_seen", time.time()
            )
        return 0.0

    def set_threshold(self, time_threshold: float) -> None:
        """Update the loitering time threshold at runtime."""
        self.time_threshold = time_threshold

    def cleanup(self, active_ids: Set[str]) -> None:
        """Remove tracker entries whose IDs are no longer active."""
        stale = [tid for tid in self._tracker if tid not in active_ids]
        for tid in stale:
            del self._tracker[tid]

    def get_all_subjects(self) -> Dict[str, dict]:
        """Return a *copy* of all tracked subject records."""
        return dict(self._tracker)
