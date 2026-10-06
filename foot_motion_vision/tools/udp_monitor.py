"""Reference UDP receiver: prints packets and measures delivery latency.

    python tools/udp_monitor.py                # pretty one-line-per-packet view
    python tools/udp_monitor.py --events-only  # only print new gesture events
    python tools/udp_monitor.py --raw          # dump decoded packets as JSON

Shows how a game client should consume the stream: decode, de-duplicate
events by ``id``, and always use the latest ``seq`` (drop out-of-order data).
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision_service.protocol import decode  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5005)
    p.add_argument("--events-only", action="store_true")
    p.add_argument("--raw", action="store_true")
    p.add_argument("--count", type=int, default=0, help="exit after N packets (0 = forever)")
    a = p.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((a.host, a.port))
    print(f"listening on udp://{a.host}:{a.port}", file=sys.stderr)

    last_seq = -1
    seen_events: deque[int] = deque(maxlen=256)
    transit_ms: deque[float] = deque(maxlen=300)
    received = lost = 0
    last_report = time.time()
    try:
        while True:
            data, _ = sock.recvfrom(4096)
            now = time.time()
            pkt = decode(data)
            received += 1
            seq = pkt["seq"]
            if seq <= last_seq:
                continue  # out of order / duplicate: stale motion data
            if last_seq >= 0:
                lost += seq - last_seq - 1
            last_seq = seq
            # Capture -> received, minus time spent inside the vision service.
            transit_ms.append((now - pkt["timestamp"]) * 1000.0 - pkt["latency_ms"])

            new_events = [e for e in pkt["events"] if e["id"] not in seen_events]
            seen_events.extend(e["id"] for e in new_events)

            if a.raw:
                print(json.dumps(pkt))
            elif new_events:
                for e in new_events:
                    print(f">>> {e['type']:<12} foot={e['foot']!s:<5} intensity={e['intensity']:.2f} x={e['x']:+.2f}")
            elif not a.events_only:
                l, r = pkt["feet_pos"]["left"], pkt["feet_pos"]["right"]
                v = pkt["velocity_y"]
                print(f"#{seq:<6} {'TRK' if pkt['tracking'] else '---'} {pkt['backend']:<9} "
                      f"L({l['x']:+.2f},{l['y']:+.2f}) R({r['x']:+.2f},{r['y']:+.2f}) "
                      f"vy L{v['left']:+.2f} R{v['right']:+.2f}  proc {pkt['latency_ms']:.1f}ms")

            if now - last_report > 5 and transit_ms:
                print(f"[stats] received={received} lost={lost} "
                      f"udp transit median={statistics.median(transit_ms):.3f}ms max={max(transit_ms):.3f}ms",
                      file=sys.stderr)
                last_report = now
            if a.count and received >= a.count:
                break
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        if transit_ms:
            print(f"[final] received={received} lost={lost} "
                  f"udp transit median={statistics.median(transit_ms):.3f}ms", file=sys.stderr)


if __name__ == "__main__":
    main()
