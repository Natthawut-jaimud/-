"""OpenCV debug view: lower-body skeleton, warnings, per-foot telemetry."""

from __future__ import annotations

import time

import cv2
import numpy as np

from .gestures import Action, FootPhase, GestureRecognizer
from .landmarks import SKELETON_EDGES, Joint
from .pipeline import FrameState, hip_missing

GREEN = (80, 220, 80)
YELLOW = (0, 220, 255)
RED = (40, 40, 230)
WHITE = (255, 255, 255)
GREY = (150, 150, 150)
CYAN = (230, 200, 40)
FONT = cv2.FONT_HERSHEY_SIMPLEX

_LEFT_COLOR = (255, 140, 40)   # player's left = blue-ish
_RIGHT_COLOR = (40, 140, 255)  # player's right = orange


class DebugOverlay:
    def __init__(self, mirror: bool = True, flash_s: float = 0.6, min_visibility: float = 0.5):
        self.mirror = mirror
        self.flash_s = flash_s
        self.min_visibility = min_visibility
        self._flash: list[tuple[str, float]] = []

    def _pt(self, p: np.ndarray, w: int) -> tuple[int, int]:
        x = w - 1 - p[0] if self.mirror else p[0]
        return int(round(x)), int(round(p[1]))

    def draw(self, image: np.ndarray, state: FrameState, recognizer: GestureRecognizer, stats: dict) -> np.ndarray:
        img = cv2.flip(image, 1) if self.mirror else image.copy()
        h, w = img.shape[:2]
        now = time.perf_counter()

        pts = state.smoothed
        if pts is not None:
            est = state.raw.estimated if state.raw is not None else np.zeros(len(pts), bool)
            for a, b in SKELETON_EDGES:
                if min(pts[a, 3], pts[b, 3]) < self.min_visibility:
                    continue
                color = _LEFT_COLOR if a.name.startswith("LEFT") else _RIGHT_COLOR
                if a == Joint.LEFT_HIP and b == Joint.RIGHT_HIP:
                    color = WHITE
                cv2.line(img, self._pt(pts[a], w), self._pt(pts[b], w), color, 3, cv2.LINE_AA)
            for j in Joint:
                if pts[j, 3] < self.min_visibility * 0.5:
                    continue
                color = GREY if est[j] else (_LEFT_COLOR if j.name.startswith("LEFT") else _RIGHT_COLOR)
                cv2.circle(img, self._pt(pts[j], w), 6, color, -1, cv2.LINE_AA)
                cv2.circle(img, self._pt(pts[j], w), 6, WHITE, 1, cv2.LINE_AA)

        # Learned floor line per foot.
        frame = state.frame
        if frame is not None and frame.valid and pts is not None:
            for side, foot in recognizer.feet.items():
                if foot.floor_px is None:
                    continue
                ankle = pts[Joint.LEFT_ANKLE if side == "left" else Joint.RIGHT_ANKLE]
                x, _ = self._pt(ankle, w)
                y = int(foot.floor_px)
                color = YELLOW if foot.phase != FootPhase.GROUNDED else GREEN
                cv2.line(img, (x - 35, y), (x + 35, y), color, 2)

        if hip_missing(state, self.min_visibility):
            self._warning(img, "HIP NOT DETECTED",
                          "Hips are the reference point: step back / raise camera tilt (5-10 deg up)")

        for e in state.events:
            self._flash.append((e.action.value, now))
        self._flash = [(a, t0) for a, t0 in self._flash if now - t0 < self.flash_s]
        for i, (a, _) in enumerate(reversed(self._flash[-3:])):
            color = RED if "STOMP" in a else (CYAN if a == Action.JUMP.value else YELLOW)
            (tw, th), _ = cv2.getTextSize(a, FONT, 1.2, 2)
            org = (w // 2 - tw // 2, h - 30 - i * 46)
            cv2.rectangle(img, (org[0] - 8, org[1] - th - 8), (org[0] + tw + 8, org[1] + 10), (0, 0, 0), -1)
            cv2.putText(img, a, org, FONT, 1.2, color, 2, cv2.LINE_AA)

        self._telemetry(img, state, recognizer, stats)
        return img

    @staticmethod
    def _warning(img: np.ndarray, title: str, hint: str) -> None:
        h, w = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), RED, -1)
        cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
        cv2.rectangle(img, (2, 2), (w - 3, h - 3), RED, 4)
        cv2.putText(img, "! " + title, (12, 32), FONT, 0.9, WHITE, 2, cv2.LINE_AA)
        cv2.putText(img, hint, (12, 58), FONT, 0.45, WHITE, 1, cv2.LINE_AA)

    @staticmethod
    def _telemetry(img: np.ndarray, state: FrameState, rec: GestureRecognizer, stats: dict) -> None:
        h, w = img.shape[:2]
        pkt = state.packet
        lines = [
            f"FPS {stats.get('fps', 0):5.1f}  proc {stats.get('proc_ms', 0):5.1f} ms  e2e {pkt.latency_ms:5.1f} ms",
            f"backend {pkt.backend}  filter {stats.get('filter', '?')}  udp {stats.get('udp', '')}",
            f"seq {pkt.seq}  tracking {'yes' if pkt.tracking else 'NO'}  scale {state.frame.scale_px if state.frame else 0:5.0f}px",
        ]
        y0 = 90
        panel = img.copy()
        cv2.rectangle(panel, (4, y0 - 16), (430, y0 + 3 * 18 - 8), (0, 0, 0), -1)
        cv2.addWeighted(panel, 0.55, img, 0.45, 0, img)
        for i, text in enumerate(lines):
            cv2.putText(img, text, (10, y0 + i * 18), FONT, 0.42, WHITE, 1, cv2.LINE_AA)

        # Foot height bars (right edge): 0..0.6 leg lengths.
        for k, (side, foot) in enumerate(rec.feet.items()):
            x = w - 70 + k * 32
            top, bottom = 90, 290
            cv2.rectangle(img, (x, top), (x + 22, bottom), GREY, 1)
            frac = float(np.clip(foot.height / 0.6, 0.0, 1.0))
            color = _LEFT_COLOR if side == "left" else _RIGHT_COLOR
            cv2.rectangle(img, (x + 2, int(bottom - frac * (bottom - top))), (x + 20, bottom - 1), color, -1)
            cv2.putText(img, side[0].upper(), (x + 5, bottom + 18), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
            cv2.putText(img, foot.phase.value[0], (x + 5, top - 8), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
            vy = -foot.v_up
            cv2.putText(img, f"{vy:+.1f}", (x - 4, bottom + 36), FONT, 0.38, WHITE, 1, cv2.LINE_AA)
