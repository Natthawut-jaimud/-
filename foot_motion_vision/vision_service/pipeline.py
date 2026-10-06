"""Camera-independent processing pipeline: pose -> smoothing -> normalisation
-> gestures -> packet.  Kept free of I/O so it can be unit-tested with a fake
pose backend and reused by other front-ends.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .filters import LandmarkSmoother, SmoothingConfig
from .gestures import Action, GestureConfig, GestureEvent, GestureRecognizer, primary_action
from .landmarks import Joint, PoseResult
from .normalization import BodyNormalizer, NormalizationConfig, NormalizedFrame
from .pose.base import PoseBackend
from .protocol import MotionPacket, Vec3


@dataclass
class PipelineConfig:
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    gestures: GestureConfig = field(default_factory=GestureConfig)
    # Each event is repeated in this many consecutive packets (UDP loss cover).
    event_repeat: int = 3
    include_joints: bool = False


@dataclass
class FrameState:
    """Everything the debug overlay needs about one processed frame."""

    raw: PoseResult | None
    smoothed: np.ndarray | None
    frame: NormalizedFrame | None
    events: list[GestureEvent]
    packet: MotionPacket


class VisionPipeline:
    def __init__(self, backend: PoseBackend, cfg: PipelineConfig | None = None):
        self.cfg = cfg or PipelineConfig()
        self.backend = backend
        self.smoother = LandmarkSmoother(self.cfg.smoothing)
        self.normalizer = BodyNormalizer(self.cfg.normalization)
        self.recognizer = GestureRecognizer(self.cfg.gestures)
        self._seq = 0
        self._recent: deque[tuple[dict, int]] = deque()

    def set_smoothing(self, method: str) -> None:
        self.cfg.smoothing.method = method
        self.smoother = LandmarkSmoother(self.cfg.smoothing)

    def reset(self) -> None:
        self.smoother.reset()
        self.normalizer.reset()
        self.recognizer.reset()

    def _active_backend(self) -> str:
        return getattr(self.backend, "active", self.backend.name)

    def process(self, image_bgr: np.ndarray, t: float, wall_time: float) -> FrameState:
        raw = self.backend.process(image_bgr, t)
        smoothed = None
        frame = None
        events: list[GestureEvent] = []
        if raw is not None:
            smoothed = self.smoother(raw.points, t)
            pose = PoseResult(smoothed, t, raw.backend, raw.image_size, raw.estimated)
            frame = self.normalizer(pose)
            events = self.recognizer.update(frame)
        else:
            # Keep the recognizer's clock running so it can time out.
            self.recognizer.update(NormalizedFrame(t=t, valid=False, scale_px=0.0, hip_center_px=np.zeros(2)))
        return FrameState(raw, smoothed, frame, events, self._make_packet(frame, events, wall_time, raw))

    def _make_packet(self, frame: NormalizedFrame | None, events: list[GestureEvent], wall_time: float,
                     raw: PoseResult | None) -> MotionPacket:
        self._seq += 1
        for e in events:
            self._recent.append((e.to_dict(), self.cfg.event_repeat))
        repeated = []
        for _ in range(len(self._recent)):
            d, n = self._recent.popleft()
            repeated.append(d)
            if n > 1:
                self._recent.append((d, n - 1))

        pkt = MotionPacket(
            seq=self._seq,
            timestamp=wall_time,
            action=primary_action(events) if events else Action.IDLE,
            tracking=bool(frame is not None and frame.valid),
            backend=raw.backend if raw is not None else self._active_backend(),
            events=repeated,
        )
        if frame is not None and frame.valid:
            la = frame.sides["left"].joints["ankle"]
            ra = frame.sides["right"].joints["ankle"]
            pkt.left = Vec3(*map(float, la))
            pkt.right = Vec3(*map(float, ra))
            pkt.velocity_left = self.recognizer.velocity_y("left")
            pkt.velocity_right = self.recognizer.velocity_y("right")
            if self.cfg.include_joints:
                pkt.joints = {
                    side: {name: [round(float(v), 4) for v in xyz] for name, xyz in s.joints.items()}
                    for side, s in frame.sides.items()
                }
        return pkt


def hip_missing(state: FrameState, threshold: float = 0.5) -> bool:
    raw = state.raw
    if raw is None:
        return True
    return not (raw.points[Joint.LEFT_HIP, 3] >= threshold and raw.points[Joint.RIGHT_HIP, 3] >= threshold)
