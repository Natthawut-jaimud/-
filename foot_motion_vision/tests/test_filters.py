import numpy as np
import pytest

from vision_service.filters import EMAFilter, LandmarkSmoother, OneEuroFilter, SmoothingConfig


def _run(f, xs, fps=60.0):
    return np.array([f(np.array([x]), i / fps)[0] for i, x in enumerate(xs)])


def test_one_euro_reduces_jitter_when_still():
    rng = np.random.default_rng(1)
    noisy = 100 + rng.normal(0, 1.5, 600)
    out = _run(OneEuroFilter(), noisy)
    assert np.std(np.diff(out[100:])) < np.std(np.diff(noisy[100:])) / 5


def test_one_euro_less_lag_than_equally_smooth_ema_on_fast_motion():
    t = np.arange(120) / 60.0
    fast = np.where(t < 1.0, 0.0, np.minimum((t - 1.0) * 1500.0, 300.0))  # 1500 px/s move
    oe = _run(OneEuroFilter(), fast)
    ema = _run(EMAFilter(alpha=0.2), fast)
    i = 70  # mid-motion
    assert abs(fast[i] - oe[i]) < abs(fast[i] - ema[i])
    # and it settles: no accumulated delay
    assert oe[-1] == pytest.approx(300.0, abs=1.0)


def test_ema_is_frame_rate_independent():
    f1 = EMAFilter(alpha=0.5)
    f1(np.array([0.0]), 0.0)
    y30 = f1(np.array([10.0]), 1 / 30)[0]
    f2 = EMAFilter(alpha=0.5)
    f2(np.array([0.0]), 0.0)
    f2(np.array([10.0]), 1 / 60)
    y60 = f2(np.array([10.0]), 2 / 60)[0]
    assert y30 == pytest.approx(y60, rel=1e-9)


def test_smoother_freezes_low_visibility_joint_and_resets_after_timeout():
    sm = LandmarkSmoother(SmoothingConfig(reset_after_s=0.2))
    pts = np.zeros((10, 4))
    pts[:, 3] = 0.9
    pts[:, 0] = 100
    sm(pts, 0.0)
    occluded = pts.copy()
    occluded[4, 0] = 400
    occluded[4, 3] = 0.1
    out = sm(occluded, 1 / 60)
    assert out[4, 0] == pytest.approx(100)  # frozen
    reappear = pts.copy()
    reappear[4, 0] = 400
    out = sm(reappear, 0.5)  # gap > reset_after_s -> snap, no slide
    assert out[4, 0] == pytest.approx(400)


@pytest.mark.parametrize("method", ["one_euro", "ema", "none"])
def test_all_methods_run(method):
    sm = LandmarkSmoother(SmoothingConfig(method=method))
    pts = np.ones((10, 4))
    for i in range(5):
        out = sm(pts * (i + 1), i / 60)
    assert out.shape == (10, 4)
