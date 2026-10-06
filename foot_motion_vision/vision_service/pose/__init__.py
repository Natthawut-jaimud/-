"""Pose estimation backends (MediaPipe primary, YOLOv8-Pose fallback)."""

from __future__ import annotations

from .base import PoseBackend
from .hybrid import HybridConfig, HybridPoseBackend


def create_backend(name: str, **kwargs) -> PoseBackend:
    """Factory: ``mediapipe`` | ``yolo`` | ``hybrid``.

    Heavy dependencies are imported lazily so the package (and its tests)
    work without ultralytics installed.
    """
    mp_kwargs = {k: kwargs[k] for k in ("variant", "model_dir", "min_detection_confidence",
                                        "min_tracking_confidence") if k in kwargs}
    yolo_kwargs = {k: kwargs[k] for k in ("yolo_model", "device", "imgsz") if k in kwargs}
    if name == "mediapipe":
        from .mediapipe_backend import MediaPipeBackend
        return MediaPipeBackend(**mp_kwargs)
    if name == "yolo":
        from .yolo_backend import YoloPoseBackend
        return YoloPoseBackend(**yolo_kwargs)
    if name == "hybrid":
        from .mediapipe_backend import MediaPipeBackend
        from .yolo_backend import YoloPoseBackend
        return HybridPoseBackend(
            MediaPipeBackend(**mp_kwargs),
            lambda: YoloPoseBackend(**yolo_kwargs),
            kwargs.get("hybrid_config") or HybridConfig(),
        )
    raise ValueError(f"unknown backend {name!r}")


__all__ = ["PoseBackend", "HybridConfig", "HybridPoseBackend", "create_backend"]
