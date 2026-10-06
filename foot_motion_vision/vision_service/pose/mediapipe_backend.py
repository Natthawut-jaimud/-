"""MediaPipe Pose (BlazePose, 33 landmarks) backend.

Uses the MediaPipe Tasks ``PoseLandmarker`` in VIDEO mode (detector runs
once, then a tracker follows the person between frames - much faster than
detecting on every frame).  Falls back to the legacy ``mp.solutions.pose``
API on old MediaPipe installs that do not ship the Tasks API.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from ..landmarks import LOWER_BODY, MEDIAPIPE_INDEX, NUM_JOINTS, PoseResult
from .base import PoseBackend

log = logging.getLogger(__name__)

MODEL_URLS = {
    v: f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task"
    for v in ("lite", "full", "heavy")
}
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


def ensure_model(variant: str, model_dir: Path | str = DEFAULT_MODEL_DIR) -> Path:
    if variant not in MODEL_URLS:
        raise ValueError(f"variant must be one of {sorted(MODEL_URLS)}")
    model_dir = Path(model_dir)
    path = model_dir / f"pose_landmarker_{variant}.task"
    if not path.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s", MODEL_URLS[variant])
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URLS[variant], tmp)
        tmp.replace(path)
    return path


class MediaPipeBackend(PoseBackend):
    name = "mediapipe"

    def __init__(
        self,
        variant: str = "full",
        model_dir: Path | str = DEFAULT_MODEL_DIR,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        import mediapipe as mp

        self._mp = mp
        self._last_ts_ms = -1
        self._legacy = None
        self._landmarker = None
        try:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions
        except ImportError:
            vision = None
        if vision is not None:
            options = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(ensure_model(variant, model_dir))),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=min_detection_confidence,
                min_pose_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
        elif hasattr(mp, "solutions"):
            complexity = {"lite": 0, "full": 1, "heavy": 2}[variant]
            self._legacy = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=complexity,
                smooth_landmarks=False,  # we run our own filter
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        else:
            raise RuntimeError("this MediaPipe build has neither the Tasks API nor mp.solutions")

    def process(self, image_bgr: np.ndarray, t: float) -> PoseResult | None:
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self._landmarker is not None:
            ts_ms = max(int(t * 1000.0), self._last_ts_ms + 1)  # must strictly increase
            self._last_ts_ms = ts_ms
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect_for_video(mp_image, ts_ms)
            if not result.pose_landmarks:
                return None
            landmarks = result.pose_landmarks[0]
        else:
            rgb.flags.writeable = False
            result = self._legacy.process(rgb)
            if result.pose_landmarks is None:
                return None
            landmarks = result.pose_landmarks.landmark

        pts = np.zeros((NUM_JOINTS, 4))
        for joint in LOWER_BODY:
            lm = landmarks[MEDIAPIPE_INDEX[joint]]
            vis = getattr(lm, "visibility", None)
            pres = getattr(lm, "presence", None)
            conf = 1.0 if vis is None else float(vis)
            if pres is not None:
                conf = min(conf, float(pres))
            # x,y normalised to [0,1]; z is on roughly the same scale as x.
            pts[joint] = (lm.x * w, lm.y * h, (lm.z or 0.0) * w, conf)
        return PoseResult(points=pts, timestamp=t, backend=self.name, image_size=(w, h))

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        if self._legacy is not None:
            self._legacy.close()
            self._legacy = None
