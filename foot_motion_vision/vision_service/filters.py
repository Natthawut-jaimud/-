"""Landmark smoothing: One Euro Filter and EMA, vectorised with numpy.

The One Euro Filter (Casiez et al., CHI 2012) is an adaptive low-pass filter:
at low speed it uses a low cutoff (kills jitter), at high speed the cutoff
rises with ``beta * |speed|`` so fast foot motion is not delayed.  That is
what makes it "lag-free" for stomps/kicks while keeping a standing foot still.

Both filters are first-order IIR filters, so there is no accumulated delay:
the state is a single value per coordinate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .landmarks import NUM_JOINTS


def _alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray | float:
    tau = 1.0 / (2.0 * math.pi * np.asarray(cutoff))
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Vectorised One Euro Filter over an array of any fixed shape.

    Parameters are in the units of the filtered signal (here: pixels).
    - ``min_cutoff`` (Hz): lower = smoother when still, more lag at slow speed.
    - ``beta``: speed coefficient; higher = less lag on fast motion.
    - ``d_cutoff`` (Hz): cutoff for the derivative estimate.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 2.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    def reset(self) -> None:
        self._x = self._dx = None
        self._t = None

    @property
    def derivative(self) -> np.ndarray | None:
        """Filtered rate of change (units/s) from the last update."""
        return self._dx

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x is None or self._t is None:
            self._x = x.copy()
            self._dx = np.zeros_like(x)
            self._t = t
            return self._x.copy()
        dt = t - self._t
        if dt <= 0.0:
            return self._x.copy()
        self._t = t
        dx = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        a = _alpha(cutoff, dt)
        self._x = a * x + (1.0 - a) * self._x
        return self._x.copy()


class EMAFilter:
    """Time-constant based exponential moving average.

    ``alpha`` is the weight at the reference rate ``ref_fps``; it is converted
    to a time constant so behaviour stays the same when the frame rate drifts.
    """

    def __init__(self, alpha: float = 0.5, ref_fps: float = 60.0):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.ref_fps = ref_fps
        self._tau = math.inf if alpha >= 1.0 else -1.0 / (ref_fps * math.log(1.0 - alpha))
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    def reset(self) -> None:
        self._x = self._dx = None
        self._t = None

    @property
    def derivative(self) -> np.ndarray | None:
        return self._dx

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x is None or self._t is None:
            self._x = x.copy()
            self._dx = np.zeros_like(x)
            self._t = t
            return self._x.copy()
        dt = t - self._t
        if dt <= 0.0:
            return self._x.copy()
        a = 1.0 if math.isinf(self._tau) else 1.0 - math.exp(-dt / self._tau)
        new = a * x + (1.0 - a) * self._x
        self._dx = (new - self._x) / dt
        self._x = new
        self._t = t
        return self._x.copy()


class PassThroughFilter:
    def __init__(self) -> None:
        self._x: np.ndarray | None = None
        self._t: float | None = None
        self._dx: np.ndarray | None = None

    def reset(self) -> None:
        self._x = self._dx = None
        self._t = None

    @property
    def derivative(self) -> np.ndarray | None:
        return self._dx

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x is not None and self._t is not None and t > self._t:
            self._dx = (x - self._x) / (t - self._t)
        elif self._dx is None:
            self._dx = np.zeros_like(x)
        self._x, self._t = x.copy(), t
        return x.copy()


@dataclass
class SmoothingConfig:
    method: str = "one_euro"  # one_euro | ema | none
    # Tuned for foot landmarks in pixels at 640x480 / 60 FPS: a still foot's
    # frame-to-frame jitter drops ~8x while a 0.05 s stomp keeps >95% of its
    # peak speed.  For 1280x720 scale beta by ~0.5 (it multiplies px/s).
    min_cutoff: float = 1.0
    beta: float = 0.05
    d_cutoff: float = 2.0
    ema_alpha: float = 0.55
    # Joints below this visibility keep their last smoothed value.
    min_visibility: float = 0.35
    # After this many seconds without a valid measurement a joint is reset
    # (so it snaps to the new position instead of sliding in from far away).
    reset_after_s: float = 0.5


def make_filter(cfg: SmoothingConfig):
    if cfg.method == "one_euro":
        return OneEuroFilter(cfg.min_cutoff, cfg.beta, cfg.d_cutoff)
    if cfg.method == "ema":
        return EMAFilter(cfg.ema_alpha)
    if cfg.method == "none":
        return PassThroughFilter()
    raise ValueError(f"unknown smoothing method: {cfg.method!r}")


class LandmarkSmoother:
    """Smooths ``(NUM_JOINTS, 4)`` landmark arrays joint by joint.

    Each joint has its own filter so that an occluded joint (low visibility)
    can be frozen / reset without disturbing the others.
    """

    def __init__(self, cfg: SmoothingConfig | None = None):
        self.cfg = cfg or SmoothingConfig()
        self._filters = [make_filter(self.cfg) for _ in range(NUM_JOINTS)]
        self._last = np.zeros((NUM_JOINTS, 4))
        self._last_seen = np.full(NUM_JOINTS, -math.inf)
        self._velocity = np.zeros((NUM_JOINTS, 3))

    def reset(self) -> None:
        for f in self._filters:
            f.reset()
        self._last[:] = 0.0
        self._last_seen[:] = -math.inf
        self._velocity[:] = 0.0

    @property
    def velocity(self) -> np.ndarray:
        """Per-joint filtered velocity (px/s), shape (NUM_JOINTS, 3)."""
        return self._velocity.copy()

    def __call__(self, points: np.ndarray, t: float) -> np.ndarray:
        out = self._last.copy()
        for j in range(NUM_JOINTS):
            vis = points[j, 3]
            if vis < self.cfg.min_visibility:
                # Keep last value but decay confidence so consumers know it is stale.
                out[j, 3] = min(self._last[j, 3], vis)
                if t - self._last_seen[j] > self.cfg.reset_after_s:
                    self._filters[j].reset()
                    self._velocity[j] = 0.0
                continue
            f = self._filters[j]
            if t - self._last_seen[j] > self.cfg.reset_after_s:
                f.reset()
            out[j, :3] = f(points[j, :3], t)
            out[j, 3] = vis
            d = f.derivative
            self._velocity[j] = d if d is not None else 0.0
            self._last_seen[j] = t
        self._last = out
        return out.copy()
