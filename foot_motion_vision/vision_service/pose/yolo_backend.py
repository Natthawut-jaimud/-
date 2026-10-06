"""YOLOv8-Pose fallback backend (requires ``pip install ultralytics``).

YOLO handles fast motion and motion blur better than BlazePose's tracker,
but it uses the COCO-17 skeleton which has no heel / foot-index points.
Those are synthesised from the shin direction and flagged in
``PoseResult.estimated`` (with reduced visibility) so downstream code can
treat them with care.
"""

from __future__ import annotations

import logging

import numpy as np

from ..landmarks import COCO_INDEX, NUM_JOINTS, SIDE_JOINTS, PoseResult
from .base import PoseBackend

log = logging.getLogger(__name__)

# Heel sits slightly below/behind the ankle, the toes further below/in front.
# With the camera at shin height, "in front" is mostly "down" in the image.
HEEL_OFFSET = 0.10   # fraction of shin length
TOE_OFFSET = 0.22
ESTIMATED_VISIBILITY_FACTOR = 0.6


class YoloPoseBackend(PoseBackend):
    name = "yolo"

    def __init__(self, yolo_model: str = "yolov8n-pose.pt", device: str | None = None, imgsz: int = 480):
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise RuntimeError("YOLO fallback needs `pip install ultralytics`") from exc
        self.model = YOLO(yolo_model)
        self.device = device
        self.imgsz = imgsz

    def process(self, image_bgr: np.ndarray, t: float) -> PoseResult | None:
        h, w = image_bgr.shape[:2]
        results = self.model.predict(image_bgr, imgsz=self.imgsz, device=self.device, verbose=False, max_det=3)
        if not results:
            return None
        r = results[0]
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            return None
        xy = r.keypoints.xy.cpu().numpy()
        conf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(xy.shape[:2])
        # Player = largest box (closest to the camera).
        boxes = r.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        i = int(np.argmax(areas))
        return coco_to_lower_body(xy[i], conf[i], t, (w, h), self.name)


def coco_to_lower_body(xy: np.ndarray, conf: np.ndarray, t: float, size: tuple[int, int], backend: str) -> PoseResult:
    pts = np.zeros((NUM_JOINTS, 4))
    estimated = np.zeros(NUM_JOINTS, dtype=bool)
    for joint, ci in COCO_INDEX.items():
        pts[joint] = (xy[ci, 0], xy[ci, 1], 0.0, conf[ci])
    for side in ("left", "right"):
        _, knee, ankle, heel, toe = SIDE_JOINTS[side]
        shin = pts[ankle, :2] - pts[knee, :2]
        length = float(np.linalg.norm(shin))
        direction = shin / length if length > 1e-6 else np.array([0.0, 1.0])
        vis = pts[ankle, 3] * ESTIMATED_VISIBILITY_FACTOR
        pts[heel, :2] = pts[ankle, :2] + direction * length * HEEL_OFFSET
        pts[toe, :2] = pts[ankle, :2] + direction * length * TOE_OFFSET
        pts[heel, 3] = pts[toe, 3] = vis
        estimated[heel] = estimated[toe] = True
    return PoseResult(points=pts, timestamp=t, backend=backend, image_size=size, estimated=estimated)
