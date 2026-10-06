"""Lower-body landmark definitions shared by every pose backend.

All backends return a ``PoseResult`` whose ``points`` array has one row per
entry of :data:`LOWER_BODY` (in that order) and four columns::

    [x_px, y_px, z_px, visibility]

``x_px``/``y_px`` are image pixel coordinates (origin top-left, y down).
``z_px`` is depth relative to the hips in the same pixel scale as x
(MediaPipe convention: smaller = closer to camera); backends without depth
report 0.  ``visibility`` is in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class Joint(IntEnum):
    """Row index inside ``PoseResult.points``."""

    LEFT_HIP = 0
    RIGHT_HIP = 1
    LEFT_KNEE = 2
    RIGHT_KNEE = 3
    LEFT_ANKLE = 4
    RIGHT_ANKLE = 5
    LEFT_HEEL = 6
    RIGHT_HEEL = 7
    LEFT_FOOT_INDEX = 8
    RIGHT_FOOT_INDEX = 9


LOWER_BODY: tuple[Joint, ...] = tuple(Joint)
NUM_JOINTS = len(LOWER_BODY)

# MediaPipe BlazePose (33-point topology) index for each Joint.
MEDIAPIPE_INDEX: dict[Joint, int] = {
    Joint.LEFT_HIP: 23,
    Joint.RIGHT_HIP: 24,
    Joint.LEFT_KNEE: 25,
    Joint.RIGHT_KNEE: 26,
    Joint.LEFT_ANKLE: 27,
    Joint.RIGHT_ANKLE: 28,
    Joint.LEFT_HEEL: 29,
    Joint.RIGHT_HEEL: 30,
    Joint.LEFT_FOOT_INDEX: 31,
    Joint.RIGHT_FOOT_INDEX: 32,
}

# COCO-17 index (YOLOv8-Pose). Heels / foot indexes do not exist in COCO.
COCO_INDEX: dict[Joint, int] = {
    Joint.LEFT_HIP: 11,
    Joint.RIGHT_HIP: 12,
    Joint.LEFT_KNEE: 13,
    Joint.RIGHT_KNEE: 14,
    Joint.LEFT_ANKLE: 15,
    Joint.RIGHT_ANKLE: 16,
}

SIDES = ("left", "right")

# (hip, knee, ankle, heel, foot_index) per side.
SIDE_JOINTS: dict[str, tuple[Joint, Joint, Joint, Joint, Joint]] = {
    "left": (Joint.LEFT_HIP, Joint.LEFT_KNEE, Joint.LEFT_ANKLE, Joint.LEFT_HEEL, Joint.LEFT_FOOT_INDEX),
    "right": (Joint.RIGHT_HIP, Joint.RIGHT_KNEE, Joint.RIGHT_ANKLE, Joint.RIGHT_HEEL, Joint.RIGHT_FOOT_INDEX),
}

SKELETON_EDGES: tuple[tuple[Joint, Joint], ...] = (
    (Joint.LEFT_HIP, Joint.RIGHT_HIP),
    (Joint.LEFT_HIP, Joint.LEFT_KNEE),
    (Joint.LEFT_KNEE, Joint.LEFT_ANKLE),
    (Joint.LEFT_ANKLE, Joint.LEFT_HEEL),
    (Joint.LEFT_HEEL, Joint.LEFT_FOOT_INDEX),
    (Joint.LEFT_ANKLE, Joint.LEFT_FOOT_INDEX),
    (Joint.RIGHT_HIP, Joint.RIGHT_KNEE),
    (Joint.RIGHT_KNEE, Joint.RIGHT_ANKLE),
    (Joint.RIGHT_ANKLE, Joint.RIGHT_HEEL),
    (Joint.RIGHT_HEEL, Joint.RIGHT_FOOT_INDEX),
    (Joint.RIGHT_ANKLE, Joint.RIGHT_FOOT_INDEX),
)


@dataclass
class PoseResult:
    """Output of a pose backend for a single frame."""

    points: np.ndarray  # shape (NUM_JOINTS, 4): x_px, y_px, z_px, visibility
    timestamp: float  # capture time, seconds (time.perf_counter domain)
    backend: str
    image_size: tuple[int, int]  # (width, height)
    # True where the backend could not observe the joint and synthesised it.
    estimated: np.ndarray = field(default_factory=lambda: np.zeros(NUM_JOINTS, dtype=bool))

    def visible(self, joint: Joint, threshold: float) -> bool:
        return bool(self.points[joint, 3] >= threshold)

    def hips_visible(self, threshold: float) -> bool:
        return self.visible(Joint.LEFT_HIP, threshold) and self.visible(Joint.RIGHT_HIP, threshold)
