import json
import socket
import statistics
import time

import pytest

from vision_service.gestures import Action
from vision_service.protocol import EVENT_SIZE, HEADER_SIZE, MotionPacket, Vec3, decode
from vision_service.transport import UdpSender


def sample_packet():
    return MotionPacket(
        seq=42, timestamp=1728144000.123, action=Action.STEP_LEFT, tracking=True, backend="mediapipe",
        latency_ms=9.5, left=Vec3(-0.15, 0.85, 0.02), right=Vec3(0.20, 0.90, -0.01),
        velocity_left=-1.45, velocity_right=0.02,
        events=[{"id": 7, "type": "STEP_LEFT", "foot": "left", "intensity": 1.3, "x": -0.15}],
    )


def test_json_matches_contract():
    d = json.loads(sample_packet().to_json())
    assert d["action"] == "STEP_LEFT"
    assert d["feet_pos"]["left"] == {"x": -0.15, "y": 0.85, "z": 0.02}
    assert d["velocity_y"] == {"left": -1.45, "right": 0.02}
    assert d["timestamp"] == pytest.approx(1728144000.123)
    assert d["events"][0]["id"] == 7


def test_binary_roundtrip():
    pkt = sample_packet()
    data = pkt.to_binary()
    assert len(data) == HEADER_SIZE + EVENT_SIZE
    d = decode(data)
    assert d["seq"] == 42 and d["action"] == "STEP_LEFT" and d["tracking"]
    assert d["timestamp"] == pytest.approx(pkt.timestamp)
    assert d["feet_pos"]["right"]["x"] == pytest.approx(0.20, abs=1e-6)
    assert d["velocity_y"]["left"] == pytest.approx(-1.45, abs=1e-6)
    assert d["events"][0] == {"id": 7, "type": "STEP_LEFT", "foot": "left",
                              "intensity": pytest.approx(1.3, abs=1e-6), "x": pytest.approx(-0.15, abs=1e-6)}


def test_json_decode_and_size():
    data = sample_packet().to_json()
    assert decode(data)["action"] == "STEP_LEFT"
    assert len(data) < 600


@pytest.mark.parametrize("fmt", ["json", "binary"])
def test_udp_localhost_latency_under_2ms(fmt):
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(1.0)
    port = rx.getsockname()[1]
    lat = []
    with UdpSender("127.0.0.1", port) as tx:
        for i in range(200):
            pkt = sample_packet()
            pkt.seq = i
            t0 = time.perf_counter()
            assert tx.send(pkt.encode(fmt))
            data, _ = rx.recvfrom(2048)
            decode(data)
            lat.append((time.perf_counter() - t0) * 1000)
    rx.close()
    assert statistics.median(lat) < 2.0


def test_sender_never_blocks_without_receiver():
    with UdpSender("127.0.0.1", 1) as tx:  # nothing listens on port 1
        t0 = time.perf_counter()
        for _ in range(500):
            tx.send(b"x" * 300)
        assert (time.perf_counter() - t0) < 0.5
