"""
SENTINEL v2.0 — Person Detection (YOLOv8)
==========================================
Thin wrapper around Ultralytics YOLOv8 that reproduces the original
detection loop *exactly*:

    results = model(frame, verbose=False)[0]
    for box, cls, score in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        if int(cls) == 0 and score.item() > confidence_threshold:
            x1, y1, x2, y2 = map(int, box.tolist())
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], score.item(), 'person'))
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ultralytics import YOLO

from config import YOLO_MODEL, CONFIDENCE_MIN, PERSON_CLASS_ID, DEVICE

# Type alias for a single detection tuple
Detection = Tuple[List[int], float, str]


class PersonDetector:
    """YOLOv8-based person detector.

    Parameters
    ----------
    model_path : str, optional
        Path to the YOLO weights file (default ``config.YOLO_MODEL``).
    confidence : float, optional
        Minimum confidence score (default ``config.CONFIDENCE_MIN``).
    device : str, optional
        ``'cuda'`` or ``'cpu'`` (default ``config.DEVICE``).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: Optional[float] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model = YOLO(model_path or YOLO_MODEL).to(device or DEVICE)
        self.confidence: float = confidence or CONFIDENCE_MIN

    # ------------------------------------------------------------------
    # Detection — logic identical to original lines 67-74
    # ------------------------------------------------------------------
    def detect(self, frame) -> List[Detection]:
        """Run YOLOv8 inference on *frame*.

        Returns
        -------
        list[tuple]
            Each element is ``([x, y, w, h], confidence, 'person')``.
            Coordinates are in pixel integers; format matches what
            DeepSORT expects.
        """
        results = self.model(frame, verbose=False)[0]
        detections: List[Detection] = []

        for box, cls, score in zip(
            results.boxes.xyxy, results.boxes.cls, results.boxes.conf
        ):
            if int(cls) == PERSON_CLASS_ID and score.item() > self.confidence:
                x1, y1, x2, y2 = map(int, box.tolist())
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], score.item(), "person"))

        return detections

    # ------------------------------------------------------------------
    # Runtime tunables
    # ------------------------------------------------------------------
    def set_confidence(self, confidence: float) -> None:
        """Update the minimum confidence threshold at runtime."""
        self.confidence = confidence
