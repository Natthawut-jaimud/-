"""Relative-motion gesture recognition: steps, stomps, kicks and jumps.

Two coordinate spaces are used on purpose:

* **Ground space** (image y of the foot vs. a learned floor line, divided by
  leg length) for anything that touches the floor: lifting, landing, jumping.
  The camera is fixed, so the floor line is stable even when the player
  crouches or the hips move.
* **Hip-relative space** (from :mod:`normalization`) for leg geometry: knee
  extension and lateral reach, used by kick detection.

All thresholds are in leg-length units (or leg lengths per second), so they
work the same for every body size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from .normalization import NormalizedFrame


class Action(str, Enum):
    IDLE = "IDLE"
    STEP_LEFT = "STEP_LEFT"
    STEP_RIGHT = "STEP_RIGHT"
    STOMP_LEFT = "STOMP_LEFT"
    STOMP_RIGHT = "STOMP_RIGHT"
    KICK_LEFT = "KICK_LEFT"
    KICK_RIGHT = "KICK_RIGHT"
    JUMP = "JUMP"


ACTION_CODES: dict[Action, int] = {a: i for i, a in enumerate(Action)}

# When several events fire in the same frame, the highest wins the
# single ``action`` field of the packet (all are still listed in ``events``).
_PRIORITY = {
    Action.JUMP: 4,
    Action.KICK_LEFT: 3,
    Action.KICK_RIGHT: 3,
    Action.STOMP_LEFT: 2,
    Action.STOMP_RIGHT: 2,
    Action.STEP_LEFT: 1,
    Action.STEP_RIGHT: 1,
    Action.IDLE: 0,
}

_STEP = {"left": Action.STEP_LEFT, "right": Action.STEP_RIGHT}
_STOMP = {"left": Action.STOMP_LEFT, "right": Action.STOMP_RIGHT}
_KICK = {"left": Action.KICK_LEFT, "right": Action.KICK_RIGHT}


@dataclass
class GestureConfig:
    # --- foot contact / step ---
    lift_height: float = 0.05          # foot above floor to count as lifted
    # A step lifts one foot relative to the other.  Requiring this difference
    # rejects whole-body bounces, camera wobble and heel raises on both feet.
    lift_differential: float = 0.6     # x lift_height
    contact_height: float = 0.03       # foot within this of floor = contact
    min_down_speed: float = 0.45       # descent speed needed before a landing
    stop_speed: float = 0.30           # |v| below this after descent = "stopped"
    contact_timeout_s: float = 0.10    # gentle landing: emit after this long in contact
    stomp_speed: float = 2.0           # peak descent speed for STOMP
    step_refractory_s: float = 0.15
    # --- kick / leg raise ---
    kick_min_height: float = 0.30      # foot raise above floor
    kick_min_up_speed: float = 1.6     # peak upward speed while raising
    kick_min_knee_angle: float = 140.0  # degrees, 180 = straight leg
    kick_lateral_reach: float = 0.55   # |ankle.x - hip.x| for side kicks
    kick_lateral_speed: float = 1.6
    # --- jump ---
    jump_hip_rise: float = 0.08        # hips above standing baseline
    jump_min_up_speed: float = 0.8     # hip upward speed
    jump_foot_height: float = 0.04     # both feet at least this high
    # --- tracking helpers ---
    velocity_tau_s: float = 0.025      # light smoothing of finite differences
    floor_adapt_rate: float = 2.0      # 1/s, re-learn floor while foot still
    still_speed: float = 0.15
    lost_reset_s: float = 0.5          # forget floor/hip baselines after this


@dataclass
class GestureEvent:
    id: int
    action: Action
    t: float
    foot: str | None = None
    intensity: float = 0.0  # peak speed in leg lengths / s
    x: float = 0.0           # foot x (hip-relative) at the moment of the event

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.action.value,
            "foot": self.foot,
            "intensity": round(self.intensity, 3),
            "x": round(self.x, 4),
        }


class _Derivative:
    """Finite difference with a short exponential smoothing."""

    def __init__(self, tau: float):
        self.tau = tau
        self.value = 0.0
        self._prev: float | None = None
        self._t: float | None = None

    def reset(self) -> None:
        self.value = 0.0
        self._prev = self._t = None

    def __call__(self, x: float, t: float) -> float:
        if self._prev is not None and self._t is not None and t > self._t:
            dt = t - self._t
            raw = (x - self._prev) / dt
            a = 1.0 - math.exp(-dt / self.tau) if self.tau > 0 else 1.0
            self.value += a * (raw - self.value)
        self._prev, self._t = x, t
        return self.value


class FootPhase(str, Enum):
    GROUNDED = "GROUNDED"
    LIFTED = "LIFTED"
    LANDING = "LANDING"  # back near floor, waiting for the stop


@dataclass
class FootTracker:
    side: str
    cfg: GestureConfig
    phase: FootPhase = FootPhase.GROUNDED
    floor_px: float | None = None
    height: float = 0.0       # leg lengths above floor (+ up)
    v_up: float = 0.0         # leg lengths / s (+ up)
    peak_height: float = 0.0
    peak_up: float = 0.0
    peak_down: float = 0.0
    v_x: float = 0.0          # hip-relative lateral speed, leg lengths / s
    peak_lateral: float = 0.0
    contact_since: float | None = None
    kicked: bool = False
    last_step_t: float = -math.inf
    _deriv: _Derivative = field(init=False)

    _x_deriv: _Derivative = field(init=False)

    def __post_init__(self) -> None:
        self._deriv = _Derivative(self.cfg.velocity_tau_s)
        self._x_deriv = _Derivative(self.cfg.velocity_tau_s)

    def reset(self) -> None:
        self.phase = FootPhase.GROUNDED
        self.floor_px = None
        self.height = self.v_up = 0.0
        self._reset_lift()
        self._deriv.reset()
        self._x_deriv.reset()

    def _reset_lift(self) -> None:
        self.peak_height = self.peak_up = self.peak_down = self.peak_lateral = 0.0
        self.contact_since = None
        self.kicked = False

    def update_kinematics(self, foot_y_px: float, ankle_x_rel: float, scale: float, t: float, dt: float) -> None:
        cfg = self.cfg
        self.v_x = self._x_deriv(ankle_x_rel, t)
        y = foot_y_px / scale  # image-down, leg units
        self.v_up = -self._deriv(y, t)
        if self.floor_px is None or foot_y_px > self.floor_px:
            # The foot can never be below the floor: push the floor down.
            self.floor_px = foot_y_px
        elif abs(self.v_up) < cfg.still_speed and self.phase == FootPhase.GROUNDED:
            a = 1.0 - math.exp(-cfg.floor_adapt_rate * dt)
            self.floor_px += a * (foot_y_px - self.floor_px)
        self.height = (self.floor_px - foot_y_px) / scale

    def step_logic(self, t: float, other_height: float, suppress_landing: bool) -> tuple[Action, float] | None:
        """Advance the lift/land state machine. Returns (action, peak_down) on landing."""
        cfg = self.cfg
        h, v = self.height, self.v_up
        if self.phase == FootPhase.GROUNDED:
            if h > cfg.lift_height and h - other_height > cfg.lift_height * cfg.lift_differential:
                self.phase = FootPhase.LIFTED
                self._reset_lift()
            else:
                return None
        self.peak_height = max(self.peak_height, h)
        self.peak_up = max(self.peak_up, v)
        self.peak_down = max(self.peak_down, -v)
        self.peak_lateral = max(self.peak_lateral, abs(self.v_x))

        if self.phase == FootPhase.LIFTED and h <= cfg.contact_height:
            self.phase = FootPhase.LANDING
            self.contact_since = t
        elif self.phase == FootPhase.LANDING and h > cfg.lift_height:
            # Bounced back up before settling: still the same lift.
            self.phase = FootPhase.LIFTED
            self.contact_since = None

        if self.phase != FootPhase.LANDING:
            return None
        assert self.contact_since is not None
        stopped = self.peak_down >= cfg.min_down_speed and abs(v) <= cfg.stop_speed
        timed_out = t - self.contact_since >= cfg.contact_timeout_s
        if not (stopped or timed_out):
            return None

        peak_down, kicked = self.peak_down, self.kicked
        self.phase = FootPhase.GROUNDED
        self._reset_lift()
        if suppress_landing or kicked or t - self.last_step_t < cfg.step_refractory_s:
            return None
        self.last_step_t = t
        action = _STOMP[self.side] if peak_down >= cfg.stomp_speed else _STEP[self.side]
        return action, peak_down


class GestureRecognizer:
    def __init__(self, cfg: GestureConfig | None = None):
        self.cfg = cfg or GestureConfig()
        self.feet = {s: FootTracker(s, self.cfg) for s in ("left", "right")}
        self._hip_deriv = _Derivative(self.cfg.velocity_tau_s)
        self.hip_floor_px: float | None = None  # standing hip height (image y)
        self.hip_height = 0.0
        self.hip_v_up = 0.0
        self.airborne = False
        self._next_id = 1
        self._last_valid_t: float | None = None
        self._t: float | None = None

    # ------------------------------------------------------------------ utils
    def reset(self) -> None:
        for f in self.feet.values():
            f.reset()
        self._hip_deriv.reset()
        self.hip_floor_px = None
        self.hip_height = self.hip_v_up = 0.0
        self.airborne = False

    def _event(self, action: Action, t: float, foot: str | None, intensity: float, x: float = 0.0) -> GestureEvent:
        e = GestureEvent(self._next_id, action, t, foot, intensity, x)
        self._next_id += 1
        return e

    def velocity_y(self, side: str) -> float:
        """Vertical foot velocity for the packet: leg lengths / s, + = DOWN."""
        return -self.feet[side].v_up

    # ----------------------------------------------------------------- update
    def update(self, frame: NormalizedFrame) -> list[GestureEvent]:
        cfg = self.cfg
        t = frame.t
        dt = 1.0 / 60.0 if self._t is None else max(t - self._t, 1e-4)
        self._t = t
        if not frame.valid or not frame.sides:
            if self._last_valid_t is not None and t - self._last_valid_t > cfg.lost_reset_s:
                self.reset()
                self._last_valid_t = None
            return []
        self._last_valid_t = t
        scale = frame.scale_px

        for side, foot in self.feet.items():
            s = frame.sides[side]
            foot.update_kinematics(s.foot_y_px, float(s.joints["ankle"][0] - s.joints["hip"][0]), scale, t, dt)
        self._update_hip(float(frame.hip_center_px[1]), scale, t, dt)

        events: list[GestureEvent] = []

        # ---- jump: both feet off the floor and the hips moving up
        left, right = self.feet["left"], self.feet["right"]
        both_up = min(left.height, right.height) >= cfg.jump_foot_height
        if not self.airborne and both_up and self.hip_height >= cfg.jump_hip_rise and self.hip_v_up >= cfg.jump_min_up_speed:
            self.airborne = True
            events.append(self._event(Action.JUMP, t, None, self.hip_v_up))
        elif self.airborne and max(left.height, right.height) <= cfg.contact_height:
            self.airborne = False

        # ---- steps / stomps on landing (also advances each foot's lift phase)
        for side, foot in self.feet.items():
            other = self.feet["right" if side == "left" else "left"]
            landed = foot.step_logic(t, other.height, suppress_landing=self.airborne)
            if landed is not None:
                action, peak_down = landed
                events.append(self._event(action, t, side, peak_down, float(frame.sides[side].joints["ankle"][0])))

        # ---- kicks / leg raises (once per lift)
        for side, foot in self.feet.items():
            if self.airborne or foot.phase != FootPhase.LIFTED or foot.kicked:
                continue
            s = frame.sides[side]
            front_kick = (
                foot.height >= cfg.kick_min_height
                and foot.peak_up >= cfg.kick_min_up_speed
                and s.knee_angle_deg >= cfg.kick_min_knee_angle
            )
            reach = abs(float(s.joints["ankle"][0] - s.joints["hip"][0]))
            side_kick = reach >= cfg.kick_lateral_reach and foot.peak_lateral >= cfg.kick_lateral_speed
            if front_kick or side_kick:
                foot.kicked = True
                intensity = max(foot.peak_up, foot.peak_lateral)
                events.append(self._event(_KICK[side], t, side, intensity, float(s.joints["ankle"][0])))
        return events

    def _update_hip(self, hip_y_px: float, scale: float, t: float, dt: float) -> None:
        cfg = self.cfg
        self.hip_v_up = -self._hip_deriv(hip_y_px / scale, t)
        grounded = all(f.phase == FootPhase.GROUNDED and f.height <= cfg.contact_height for f in self.feet.values())
        if self.hip_floor_px is None:
            self.hip_floor_px = hip_y_px
        elif grounded and not self.airborne and abs(self.hip_v_up) < cfg.still_speed:
            a = 1.0 - math.exp(-cfg.floor_adapt_rate * dt)
            self.hip_floor_px += a * (hip_y_px - self.hip_floor_px)
        self.hip_height = (self.hip_floor_px - hip_y_px) / scale


def primary_action(events: list[GestureEvent]) -> Action:
    if not events:
        return Action.IDLE
    return max(events, key=lambda e: _PRIORITY[e.action]).action
