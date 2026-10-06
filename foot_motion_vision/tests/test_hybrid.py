import numpy as np

from tests.synth import standing
from vision_service.landmarks import PoseResult
from vision_service.pose.base import PoseBackend
from vision_service.pose.hybrid import HybridConfig, HybridPoseBackend


class Fake(PoseBackend):
    def __init__(self, name, script):
        self.name = name
        self.script = script
        self.calls = 0

    def process(self, image_bgr, t):
        self.calls += 1
        pts = self.script(t)
        return None if pts is None else PoseResult(pts, t, self.name, (640, 480))


IMG = np.zeros((2, 2, 3), np.uint8)


def test_switches_to_fallback_when_primary_loses_person_and_returns():
    primary = Fake("mediapipe", lambda t: None if 0.2 <= t < 0.5 else standing())
    fallback = Fake("yolo", lambda t: standing())
    hb = HybridPoseBackend(primary, lambda: fallback, HybridConfig(miss_frames=3, hold_s=0.3))
    used = []
    for i in range(60):
        r = hb.process(IMG, i / 60)
        used.append(r.backend if r else None)
    assert used[0] == "mediapipe"
    assert "yolo" in used
    assert used[-1] == "mediapipe"
    assert hb.switches == 2


def test_high_speed_triggers_fallback():
    def prim(t):
        p = standing()
        if t >= 0.3:
            p[4:10:2, 0] += 400 * np.sin(t * 60)  # violent left-foot motion
        return p

    primary = Fake("mediapipe", prim)
    fallback = Fake("yolo", lambda t: standing())
    hb = HybridPoseBackend(primary, lambda: fallback, HybridConfig(high_speed=12.0, miss_frames=2))
    names = [hb.process(IMG, i / 60).backend for i in range(40)]
    assert names[:18] == ["mediapipe"] * 18
    assert "yolo" in names[18:]


def test_missing_fallback_dependency_keeps_primary():
    primary = Fake("mediapipe", lambda t: None)

    def broken():
        raise RuntimeError("ultralytics not installed")

    hb = HybridPoseBackend(primary, broken)
    for i in range(10):
        assert hb.process(IMG, i / 60) is None
    assert hb.active == "mediapipe"
