"""Data-streaming contract between the Vision Service and the game client.

Two encodings carry the same information:

JSON (default, easy to debug, ~300-450 bytes)::

    {
      "v": 1, "seq": 1234, "timestamp": 1728144000.123,
      "action": "STEP_LEFT",
      "tracking": true, "backend": "mediapipe", "latency_ms": 9.8,
      "feet_pos": {"left": {"x": -0.15, "y": 0.85, "z": 0.02},
                   "right": {"x": 0.20, "y": 0.90, "z": -0.01}},
      "velocity_y": {"left": -1.45, "right": 0.02},
      "events": [{"id": 57, "type": "STEP_LEFT", "foot": "left",
                  "intensity": 1.3, "x": -0.15}]
    }

Binary (fixed little-endian header of :data:`HEADER_SIZE` bytes followed by
``event_count`` records of :data:`EVENT_SIZE` bytes), see :data:`HEADER_FMT`.

``action`` holds the most important event *created in this frame* (or
``IDLE``).  Because UDP can drop datagrams, every event is additionally
repeated in ``events`` for a few consecutive packets; clients must
de-duplicate by ``id`` (monotonically increasing).
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field

from .gestures import ACTION_CODES, Action

PROTOCOL_VERSION = 1
MAGIC = b"LBMT"  # Lower-Body Motion Tracking

# magic, version, flags, action, event_count, seq, timestamp,
# left xyz, right xyz, velocity_y left/right, latency_ms
HEADER_FMT = "<4sBBBBId3f3f2ff"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
# id, action code, foot (0 none / 1 left / 2 right), intensity, x
EVENT_FMT = "<IBBff"
EVENT_SIZE = struct.calcsize(EVENT_FMT)

FLAG_TRACKING = 0x01
FLAG_FALLBACK_BACKEND = 0x02

_FOOT_CODE = {None: 0, "left": 1, "right": 2}
_FOOT_NAME = {v: k for k, v in _FOOT_CODE.items()}
_ACTION_BY_CODE = {v: k for k, v in ACTION_CODES.items()}


@dataclass
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_dict(self, nd: int = 4) -> dict:
        return {"x": round(self.x, nd), "y": round(self.y, nd), "z": round(self.z, nd)}


@dataclass
class MotionPacket:
    seq: int
    timestamp: float  # wall clock (time.time()) of frame capture
    action: Action = Action.IDLE
    tracking: bool = False
    backend: str = "none"
    latency_ms: float = 0.0  # capture -> send, inside the vision service
    left: Vec3 = field(default_factory=Vec3)
    right: Vec3 = field(default_factory=Vec3)
    velocity_left: float = 0.0
    velocity_right: float = 0.0
    events: list[dict] = field(default_factory=list)
    joints: dict | None = None  # optional full lower-body joints (debug)

    # ------------------------------------------------------------- JSON
    def to_dict(self) -> dict:
        d = {
            "v": PROTOCOL_VERSION,
            "seq": self.seq,
            "timestamp": round(self.timestamp, 4),
            "action": self.action.value,
            "tracking": self.tracking,
            "backend": self.backend,
            "latency_ms": round(self.latency_ms, 2),
            "feet_pos": {"left": self.left.to_dict(), "right": self.right.to_dict()},
            "velocity_y": {"left": round(self.velocity_left, 3), "right": round(self.velocity_right, 3)},
            "events": self.events,
        }
        if self.joints is not None:
            d["joints"] = self.joints
        return d

    def to_json(self) -> bytes:
        return json.dumps(self.to_dict(), separators=(",", ":")).encode("utf-8")

    # ----------------------------------------------------------- binary
    def to_binary(self) -> bytes:
        flags = (FLAG_TRACKING if self.tracking else 0) | (FLAG_FALLBACK_BACKEND if self.backend == "yolo" else 0)
        events = self.events[:255]
        out = [
            struct.pack(
                HEADER_FMT,
                MAGIC,
                PROTOCOL_VERSION,
                flags,
                ACTION_CODES[self.action],
                len(events),
                self.seq & 0xFFFFFFFF,
                self.timestamp,
                self.left.x, self.left.y, self.left.z,
                self.right.x, self.right.y, self.right.z,
                self.velocity_left, self.velocity_right,
                self.latency_ms,
            )
        ]
        for e in events:
            out.append(
                struct.pack(
                    EVENT_FMT,
                    e["id"] & 0xFFFFFFFF,
                    ACTION_CODES[Action(e["type"])],
                    _FOOT_CODE[e.get("foot")],
                    e.get("intensity", 0.0),
                    e.get("x", 0.0),
                )
            )
        return b"".join(out)

    def encode(self, fmt: str) -> bytes:
        if fmt == "json":
            return self.to_json()
        if fmt == "binary":
            return self.to_binary()
        raise ValueError(f"unknown packet format {fmt!r}")


def decode(data: bytes) -> dict:
    """Decode a JSON or binary packet into the JSON-shaped dict (reference client)."""
    if data[:4] == MAGIC:
        return decode_binary(data)
    return json.loads(data.decode("utf-8"))


def decode_binary(data: bytes) -> dict:
    if len(data) < HEADER_SIZE:
        raise ValueError("packet too short")
    (magic, version, flags, action, n_events, seq, ts,
     lx, ly, lz, rx, ry, rz, vl, vr, latency) = struct.unpack_from(HEADER_FMT, data, 0)
    if magic != MAGIC:
        raise ValueError("bad magic")
    if len(data) < HEADER_SIZE + n_events * EVENT_SIZE:
        raise ValueError("truncated event list")
    events = []
    for i in range(n_events):
        eid, code, foot, intensity, x = struct.unpack_from(EVENT_FMT, data, HEADER_SIZE + i * EVENT_SIZE)
        events.append({"id": eid, "type": _ACTION_BY_CODE[code].value, "foot": _FOOT_NAME[foot],
                       "intensity": intensity, "x": x})
    return {
        "v": version,
        "seq": seq,
        "timestamp": ts,
        "action": _ACTION_BY_CODE[action].value,
        "tracking": bool(flags & FLAG_TRACKING),
        "backend": "yolo" if flags & FLAG_FALLBACK_BACKEND else "mediapipe",
        "latency_ms": latency,
        "feet_pos": {"left": {"x": lx, "y": ly, "z": lz}, "right": {"x": rx, "y": ry, "z": rz}},
        "velocity_y": {"left": vl, "right": vr},
        "events": events,
    }
