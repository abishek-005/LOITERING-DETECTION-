"""
SENTINEL v2.0 — Person Tracking (DeepSORT)
==========================================
Wraps ``deep_sort_realtime.DeepSort`` with the same initialisation
parameters used in the original script:

    tracker = DeepSort(max_age=30, n_init=3, embedder='mobilenet', embedder_gpu=True)
"""

from __future__ import annotations

from typing import List, Optional

from deep_sort_realtime.deepsort_tracker import DeepSort

from config import DEEPSORT_MAX_AGE, DEEPSORT_N_INIT, USE_GPU


class PersonTracker:
    """DeepSORT multi-object tracker for person re-identification.

    Parameters
    ----------
    max_age : int, optional
        Maximum frames a track is kept without detections
        (default ``config.DEEPSORT_MAX_AGE``).
    n_init : int, optional
        Number of consecutive detections before a track is confirmed
        (default ``config.DEEPSORT_N_INIT``).
    use_gpu : bool, optional
        Whether to run the MobileNet embedder on GPU
        (default ``config.USE_GPU``).
    """

    def __init__(
        self,
        max_age: Optional[int] = None,
        n_init: Optional[int] = None,
        use_gpu: Optional[bool] = None,
    ) -> None:
        gpu = use_gpu if use_gpu is not None else USE_GPU
        self.tracker = DeepSort(
            max_age=max_age or DEEPSORT_MAX_AGE,
            n_init=n_init or DEEPSORT_N_INIT,
            embedder="mobilenet",
            embedder_gpu=gpu,
        )

    # ------------------------------------------------------------------
    # Update — mirrors original lines 76-81
    # ------------------------------------------------------------------
    def update(self, detections: list, frame) -> List:
        """Feed new detections and return only *confirmed* tracks.

        Parameters
        ----------
        detections : list
            Output of :meth:`PersonDetector.detect` — list of
            ``([x, y, w, h], confidence, 'person')`` tuples.
        frame : numpy.ndarray
            Current video frame (used by the embedder).

        Returns
        -------
        list
            Confirmed ``Track`` objects from DeepSORT.
        """
        tracks = self.tracker.update_tracks(detections, frame=frame)
        return [t for t in tracks if t.is_confirmed()]
