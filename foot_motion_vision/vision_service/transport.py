"""Non-blocking UDP sender.

``sendto`` on a non-blocking datagram socket returns immediately (a few
microseconds on localhost), so the vision loop is never blocked by the
receiver.  If nobody is listening the packet is simply dropped.
"""

from __future__ import annotations

import logging
import socket
import time

log = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5005
# Stay below a typical 1500-byte Ethernet MTU to avoid IP fragmentation.
MAX_DATAGRAM = 1400


class UdpSender:
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, broadcast: bool = False):
        self.address = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        if broadcast:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        # Small send buffer: stale motion data is worthless, never queue it.
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 64 * 1024)
        except OSError:
            pass
        self.sent = 0
        self.dropped = 0
        self.last_send_us = 0.0

    def send(self, payload: bytes) -> bool:
        if len(payload) > MAX_DATAGRAM:
            log.warning("packet of %d bytes exceeds %d, may fragment", len(payload), MAX_DATAGRAM)
        t0 = time.perf_counter()
        try:
            self.sock.sendto(payload, self.address)
        except (BlockingIOError, ConnectionRefusedError, ConnectionResetError, OSError) as exc:
            # ConnectionRefused/Reset show up on Windows after an ICMP
            # "port unreachable" when the game is not running yet.
            self.dropped += 1
            if self.dropped in (1, 100) or self.dropped % 1000 == 0:
                log.debug("UDP send dropped (%s), total dropped=%d", exc, self.dropped)
            return False
        finally:
            self.last_send_us = (time.perf_counter() - t0) * 1e6
        self.sent += 1
        return True

    def close(self) -> None:
        self.sock.close()

    def __enter__(self) -> "UdpSender":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
