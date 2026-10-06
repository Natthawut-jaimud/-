"""Synthetic lower-body skeletons (camera facing the player, image y down)."""

from __future__ import annotations

import numpy as np

from vision_service.landmarks import NUM_JOINTS, Joint, PoseResult
from vision_service.pose.base import PoseBackend

LEG = 250.0  # px, hip -> ankle along the leg when standing


def standing(scale: float = 1.0, cx: float = 320.0, hip_y: float = 150.0) -> np.ndarray:
    """Player's LEFT leg appears on the image RIGHT (larger x)."""
    p = np.zeros((NUM_JOINTS, 4))
    half = 25.0 * scale
    for side, sx in (("LEFT", +1), ("RIGHT", -1)):
        x = cx + sx * half
        p[Joint[f"{side}_HIP"]] = (x, hip_y, 0, 0.99)
        p[Joint[f"{side}_KNEE"]] = (x, hip_y + 0.5 * LEG * scale, 0, 0.99)
        p[Joint[f"{side}_ANKLE"]] = (x, hip_y + LEG * scale, 0, 0.99)
        p[Joint[f"{side}_HEEL"]] = (x, hip_y + (LEG + 10) * scale, 0, 0.95)
        p[Joint[f"{side}_FOOT_INDEX"]] = (x, hip_y + (LEG + 18) * scale, -20 * scale, 0.95)
    return p


def lift_foot(p: np.ndarray, side: str, height_legs: float, scale: float = 1.0, straight: bool = False) -> np.ndarray:
    """Raise one foot by ``height_legs`` leg lengths."""
    p = p.copy()
    s = side.upper()
    dy = height_legs * LEG * scale
    for j in ("ANKLE", "HEEL", "FOOT_INDEX"):
        p[Joint[f"{s}_{j}"], 1] -= dy
    hip, knee, ankle = Joint[f"{s}_HIP"], Joint[f"{s}_KNEE"], Joint[f"{s}_ANKLE"]
    if straight:
        p[knee, :3] = (p[hip, :3] + p[ankle, :3]) / 2.0  # collinear -> straight leg
    else:
        # Bent knee comes toward the camera (negative z).
        p[knee, 1] -= dy * 0.5
        p[knee, 2] = -dy * 1.2
    return p


def shift(p: np.ndarray, dy: float) -> np.ndarray:
    p = p.copy()
    p[:, 1] += dy
    return p


def run_profile(profile, fps: float = 60.0):
    """profile(t) -> points; yields (t, points)."""
    t = 0.0
    while True:
        pts = profile(t)
        if pts is None:
            return
        yield t, pts
        t += 1.0 / fps


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


class ScriptedBackend(PoseBackend):
    """Pose backend that replays prepared arrays (``None`` = no person)."""

    name = "scripted"

    def __init__(self, frames, noise: float = 0.0, seed: int = 0):
        self.frames = list(frames)
        self.i = 0
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def process(self, image_bgr, t):
        pts = self.frames[self.i]
        self.i += 1
        if pts is None:
            return None
        pts = pts.copy()
        if self.noise:
            pts[:, :2] += self.rng.normal(0.0, self.noise, size=(len(pts), 2))
        return PoseResult(pts, t, self.name, (640, 480))
