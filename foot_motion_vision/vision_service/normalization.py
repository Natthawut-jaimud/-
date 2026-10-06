"""Hip-relative, body-size-invariant coordinates.

Coordinate system sent to the game (player-centric, "leg length" units):

* origin  : mid-point between the two hips
* +x      : the player's RIGHT side (camera faces the player, so the raw
            image is mirrored relative to the player; we undo that here)
* +y      : DOWN (toward the floor) - a standing ankle is at y ~= +1.0
* +z      : AWAY from the camera (MediaPipe convention; 0 for YOLO)
* unit    : 1.0 == the player's leg length (hip->knee + knee->ankle)

Because everything is divided by the player's own leg length, a tall adult
and a small child produce the same numbers for the same movement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .landmarks import SIDE_JOINTS, SIDES, Joint, PoseResult


@dataclass
class NormalizationConfig:
    min_visibility: float = 0.5
    # Leg segments only look shorter under projection (foreshortening),
    # so we grow the scale quickly and shrink it slowly.
    scale_rise_rate: float = 8.0   # 1/s
    scale_fall_rate: float = 0.4   # 1/s
    # Used only when no full leg is visible: leg_length ~= hip_width * ratio.
    hip_width_to_leg_ratio: float = 4.5
    min_scale_px: float = 20.0


@dataclass
class SideState:
    """Per-leg data derived for a single frame."""

    joints: dict[str, np.ndarray]  # name -> (x, y, z) hip-relative, leg units
    foot_y_px: float                # lowest image y of ankle/heel (ground contact)
    foot_visible: bool
    knee_angle_deg: float           # 180 == straight leg
    hip_flexion_deg: float          # angle of thigh from vertical-down


@dataclass
class NormalizedFrame:
    t: float
    valid: bool                    # False when hips are not visible
    scale_px: float
    hip_center_px: np.ndarray      # (x, y)
    sides: dict[str, SideState] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC in degrees."""
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-9:
        return 180.0
    cos = float(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))
    return math.degrees(math.acos(cos))


class BodyNormalizer:
    def __init__(self, cfg: NormalizationConfig | None = None):
        self.cfg = cfg or NormalizationConfig()
        self._scale: float | None = None
        self._t: float | None = None

    @property
    def scale_px(self) -> float | None:
        return self._scale

    def reset(self) -> None:
        self._scale = None
        self._t = None

    def _measure_scale(self, p: np.ndarray) -> float | None:
        thr = self.cfg.min_visibility
        lengths = []
        for side in SIDES:
            hip, knee, ankle, _, _ = SIDE_JOINTS[side]
            if min(p[hip, 3], p[knee, 3], p[ankle, 3]) >= thr:
                thigh = np.linalg.norm(p[hip, :2] - p[knee, :2])
                shin = np.linalg.norm(p[knee, :2] - p[ankle, :2])
                lengths.append(thigh + shin)
        if lengths:
            return float(max(lengths))
        if min(p[Joint.LEFT_HIP, 3], p[Joint.RIGHT_HIP, 3]) >= thr:
            hw = float(np.linalg.norm(p[Joint.LEFT_HIP, :2] - p[Joint.RIGHT_HIP, :2]))
            return hw * self.cfg.hip_width_to_leg_ratio
        return None

    def _update_scale(self, measured: float | None, t: float) -> float | None:
        if measured is None:
            return self._scale
        measured = max(measured, self.cfg.min_scale_px)
        if self._scale is None or self._t is None:
            self._scale = measured
        else:
            dt = max(t - self._t, 1e-3)
            rate = self.cfg.scale_rise_rate if measured > self._scale else self.cfg.scale_fall_rate
            a = 1.0 - math.exp(-rate * dt)
            self._scale += a * (measured - self._scale)
        return self._scale

    def __call__(self, pose: PoseResult) -> NormalizedFrame:
        p = pose.points
        t = pose.timestamp
        thr = self.cfg.min_visibility
        scale = self._update_scale(self._measure_scale(p), t)
        self._t = t

        hips_ok = pose.hips_visible(thr)
        frame = NormalizedFrame(
            t=t,
            valid=bool(hips_ok and scale is not None),
            scale_px=float(scale or 0.0),
            hip_center_px=(p[Joint.LEFT_HIP, :2] + p[Joint.RIGHT_HIP, :2]) / 2.0,
        )
        if not hips_ok:
            frame.warnings.append("HIP_NOT_DETECTED")
        if not frame.valid:
            return frame

        hip_c = (p[Joint.LEFT_HIP, :3] + p[Joint.RIGHT_HIP, :3]) / 2.0
        # Image -> player-centric axes: flip x (camera faces player).
        axis = np.array([-1.0, 1.0, 1.0])

        def rel(j: Joint) -> np.ndarray:
            return (p[j, :3] - hip_c) * axis / scale

        for side in SIDES:
            hip, knee, ankle, heel, toe = SIDE_JOINTS[side]
            joints = {
                "hip": rel(hip),
                "knee": rel(knee),
                "ankle": rel(ankle),
                "heel": rel(heel),
                "foot_index": rel(toe),
            }
            ys = [p[j, 1] for j in (ankle, heel) if p[j, 3] >= thr]
            foot_visible = bool(ys)
            foot_y = float(max(ys)) if ys else float(p[ankle, 1])
            knee_angle = _angle(p[hip, :3], p[knee, :3], p[ankle, :3])
            thigh = p[knee, :2] - p[hip, :2]
            hip_flex = math.degrees(math.atan2(abs(thigh[0]), thigh[1]))
            frame.sides[side] = SideState(
                joints=joints,
                foot_y_px=foot_y,
                foot_visible=foot_visible,
                knee_angle_deg=knee_angle,
                hip_flexion_deg=hip_flex,
            )
        return frame
