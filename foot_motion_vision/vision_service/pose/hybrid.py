"""Primary/fallback pose backend switcher.

MediaPipe is the primary model (it has heels and toes).  When it struggles -
the person is lost for a few frames, or the feet move faster than its
tracker can follow - the YOLOv8-Pose fallback takes over for at least
``hold_s`` seconds.  After that MediaPipe is retried each frame and resumes
as soon as it delivers a confident, calm result.

The fallback is created lazily on first use, so a machine without
ultralytics still runs (it just never switches).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..landmarks import Joint, PoseResult
from .base import PoseBackend

log = logging.getLogger(__name__)

_FEET = (Joint.LEFT_ANKLE, Joint.RIGHT_ANKLE)


@dataclass
class HybridConfig:
    min_hip_visibility: float = 0.5
    miss_frames: int = 3              # consecutive bad primary frames before switching
    # Ankle speed (in hip-widths/s) above which BlazePose tends to lag/lose
    # the foot; hip width keeps this independent of distance to camera.
    high_speed: float = 12.0
    hold_s: float = 0.6               # minimum time to stay on the fallback


class HybridPoseBackend(PoseBackend):
    name = "hybrid"

    def __init__(self, primary: PoseBackend, fallback_factory: Callable[[], PoseBackend], cfg: HybridConfig | None = None):
        self.primary = primary
        self._fallback_factory = fallback_factory
        self._fallback: PoseBackend | None = None
        self._fallback_failed = False
        self.cfg = cfg or HybridConfig()
        self.active = primary.name
        self._bad = 0
        self._fallback_since = 0.0
        self._prev: PoseResult | None = None
        self.switches = 0

    # ---------------------------------------------------------------- helpers
    def _get_fallback(self) -> PoseBackend | None:
        if self._fallback is None and not self._fallback_failed:
            try:
                self._fallback = self._fallback_factory()
            except Exception as exc:  # missing ultralytics, no weights, ...
                log.warning("fallback backend unavailable (%s); staying on %s", exc, self.primary.name)
                self._fallback_failed = True
        return self._fallback

    def _foot_speed(self, r: PoseResult) -> float:
        """Max ankle speed in hip-widths per second relative to previous frame."""
        prev = self._prev
        if prev is None or r.timestamp <= prev.timestamp:
            return 0.0
        hip_w = float(np.linalg.norm(r.points[Joint.LEFT_HIP, :2] - r.points[Joint.RIGHT_HIP, :2]))
        if hip_w < 1.0:
            return 0.0
        dt = r.timestamp - prev.timestamp
        d = max(float(np.linalg.norm(r.points[j, :2] - prev.points[j, :2])) for j in _FEET)
        return d / hip_w / dt

    def _good(self, r: PoseResult | None) -> bool:
        return r is not None and r.hips_visible(self.cfg.min_hip_visibility)

    def _switch(self, name: str, t: float) -> None:
        if name != self.active:
            log.info("pose backend: %s -> %s", self.active, name)
            self.active = name
            self.switches += 1
            if name != self.primary.name:
                self._fallback_since = t

    # ---------------------------------------------------------------- process
    def process(self, image_bgr: np.ndarray, t: float) -> PoseResult | None:
        cfg = self.cfg
        on_primary = self.active == self.primary.name
        r: PoseResult | None = None
        if on_primary or t - self._fallback_since >= cfg.hold_s:
            r = self.primary.process(image_bgr, t)
            if self._good(r) and self._foot_speed(r) < cfg.high_speed:
                self._bad = 0
                self._switch(self.primary.name, t)
                self._prev = r
                return r
            self._bad += 1
            if on_primary and self._bad < cfg.miss_frames:
                # One bad/fast frame is not enough to switch models.
                self._prev = r or self._prev
                return r

        fallback = self._get_fallback()
        if fallback is None:
            self._prev = r or self._prev
            return r
        r = fallback.process(image_bgr, t)
        self._switch(fallback.name, t)
        self._prev = r or self._prev
        return r

    def close(self) -> None:
        self.primary.close()
        if self._fallback is not None:
            self._fallback.close()
