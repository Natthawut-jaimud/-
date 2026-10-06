import numpy as np
import pytest

from tests.synth import ScriptedBackend, lift_foot, shift, smoothstep, standing
from vision_service.gestures import Action
from vision_service.pipeline import VisionPipeline

FPS = 60.0
DUMMY = np.zeros((4, 4, 3), np.uint8)


def simulate(frames, noise=0.0, seed=0):
    pipe = VisionPipeline(ScriptedBackend(frames, noise=noise, seed=seed))
    events, packets = [], []
    for i in range(len(frames)):
        t = i / FPS
        st = pipe.process(DUMMY, t, 1_700_000_000.0 + t)
        events += st.events
        packets.append(st.packet)
    return events, packets


def hold(pts, seconds):
    return [pts] * int(seconds * FPS)


def ramp(f, seconds):
    n = max(int(seconds * FPS), 1)
    return [f(smoothstep((i + 1) / n)) for i in range(n)]


def step_frames(side="left", height=0.15, up_s=0.15, down_s=0.15, scale=1.0, settle=0.3):
    base = standing(scale)
    frames = hold(base, 0.5)
    frames += ramp(lambda a: lift_foot(base, side, a * height, scale), up_s)
    frames += ramp(lambda a: lift_foot(base, side, (1 - a) * height, scale), down_s)
    frames += hold(base, settle)
    return frames


def kinds(events):
    return [e.action for e in events]


def test_idle_with_jitter_produces_no_events():
    events, packets = simulate(hold(standing(), 3.0), noise=1.5)
    assert events == []
    assert all(p.action == Action.IDLE for p in packets)
    assert packets[-1].tracking


@pytest.mark.parametrize("side,expected", [("left", Action.STEP_LEFT), ("right", Action.STEP_RIGHT)])
def test_step(side, expected):
    events, _ = simulate(step_frames(side), noise=0.8)
    assert kinds(events) == [expected]
    assert events[0].foot == side


def test_fast_drop_is_stomp():
    events, _ = simulate(step_frames("right", height=0.2, up_s=0.2, down_s=0.05))
    assert kinds(events) == [Action.STOMP_RIGHT]
    assert events[0].intensity >= 2.0


def test_step_detected_on_contact_with_low_latency():
    frames = step_frames("left", height=0.15, up_s=0.15, down_s=0.1)
    contact_index = int(0.5 * FPS) + int(0.15 * FPS) + int(0.1 * FPS)
    events, _ = simulate(frames)
    assert len(events) == 1
    delay = events[0].t - contact_index / FPS
    assert delay <= 0.1  # within ~6 frames of touchdown


def test_body_size_invariance():
    small, _ = simulate(step_frames("left", scale=0.55))
    tall, _ = simulate(step_frames("left", scale=1.3))
    assert kinds(small) == kinds(tall) == [Action.STEP_LEFT]


def test_front_kick_is_kick_not_step():
    base = standing()
    frames = hold(base, 0.5)
    frames += ramp(lambda a: lift_foot(base, "right", a * 0.45, straight=True), 0.15)
    frames += hold(lift_foot(base, "right", 0.45, straight=True), 0.1)
    frames += ramp(lambda a: lift_foot(base, "right", (1 - a) * 0.45, straight=True), 0.3)
    frames += hold(base, 0.3)
    events, _ = simulate(frames)
    assert kinds(events) == [Action.KICK_RIGHT]


def test_high_knee_raise_with_bent_knee_is_a_step_not_kick():
    base = standing()
    frames = hold(base, 0.5)
    frames += ramp(lambda a: lift_foot(base, "left", a * 0.4), 0.15)
    frames += ramp(lambda a: lift_foot(base, "left", (1 - a) * 0.4), 0.2)
    frames += hold(base, 0.3)
    events, _ = simulate(frames)
    assert kinds(events) in ([Action.STEP_LEFT], [Action.STOMP_LEFT])


def test_side_kick():
    base = standing()

    def side_kick(a):
        p = lift_foot(base, "left", a * 0.25, straight=True)
        for j in (4, 6, 8):  # left ankle/heel/toe move to the image right (player's left)
            p[j, 0] += a * 0.7 * 250
        p[2, 0] += a * 0.35 * 250
        return p

    frames = hold(base, 0.5) + ramp(side_kick, 0.15) + ramp(lambda a: side_kick(1 - a), 0.3) + hold(base, 0.3)
    events, _ = simulate(frames)
    assert kinds(events) == [Action.KICK_LEFT]
    assert events[0].x < 0  # player's left is negative x


def test_jump_single_event_no_steps():
    base = standing()
    rise = 0.18 * 250
    frames = hold(base, 0.6)
    frames += ramp(lambda a: shift(base, -a * rise), 0.18)
    frames += ramp(lambda a: shift(base, -(1 - a) * rise), 0.18)
    frames += hold(base, 0.4)
    events, _ = simulate(frames, noise=0.5)
    assert kinds(events) == [Action.JUMP]


def test_crouch_is_not_a_jump_or_step():
    base = standing()

    def crouch(a):
        p = base.copy()
        for j in range(4):  # hips + knees go down, feet stay
            p[j, 1] += a * 60 * (1.0 if j < 2 else 0.5)
        return p

    frames = hold(base, 0.5) + ramp(crouch, 0.3) + hold(crouch(1), 0.3) + ramp(lambda a: crouch(1 - a), 0.3) + hold(base, 0.3)
    events, _ = simulate(frames)
    assert events == []


def test_missing_hips_reports_not_tracking():
    base = standing()
    no_hips = base.copy()
    no_hips[0, 3] = no_hips[1, 3] = 0.05
    events, packets = simulate(hold(base, 0.3) + hold(no_hips, 0.3) + [None] * 5)
    assert events == []
    assert packets[10].tracking
    assert not packets[-6].tracking and not packets[-1].tracking


def test_packet_coordinates_hip_relative():
    _, packets = simulate(hold(standing(), 0.5))
    p = packets[-1]
    assert p.left.x < 0 < p.right.x
    assert p.left.y == pytest.approx(1.0, abs=0.02)
    assert abs(p.velocity_left) < 0.05


def test_events_repeated_for_udp_loss():
    events, packets = simulate(step_frames("left"))
    eid = events[0].id
    carrying = [p for p in packets if any(e["id"] == eid for e in p.events)]
    assert len(carrying) == 3
    assert carrying[0].action == Action.STEP_LEFT
    assert carrying[1].action == Action.IDLE


def test_whole_body_bounce_is_not_a_step():
    base = standing()
    frames = hold(base, 0.5) + [shift(base, -20 * abs(np.sin(i / 4))) for i in range(120)] + hold(base, 0.3)
    events, _ = simulate(frames, noise=0.5)
    assert all(e.action == Action.JUMP for e in events)  # never STEP_*
