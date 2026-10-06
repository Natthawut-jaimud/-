"""The running Vision Service: camera -> pipeline -> UDP (+ optional debug view)."""

from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass, field

import cv2

from .capture import CameraConfig, CameraStream
from .pipeline import PipelineConfig, VisionPipeline
from .pose import create_backend
from .transport import DEFAULT_HOST, DEFAULT_PORT, UdpSender

log = logging.getLogger(__name__)

_FILTER_CYCLE = ("one_euro", "ema", "none")


@dataclass
class ServiceConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    backend: str = "mediapipe"         # mediapipe | yolo | hybrid
    backend_kwargs: dict = field(default_factory=dict)
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    packet_format: str = "json"        # json | binary
    show_debug: bool = True
    mirror_view: bool = True
    stats_interval_s: float = 5.0
    max_frames: int | None = None      # stop after N frames (testing)


class _RateMeter:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.fps = 0.0
        self.proc_ms = 0.0
        self._last: float | None = None

    def tick(self, t: float, proc_ms: float) -> None:
        if self._last is not None and t > self._last:
            inst = 1.0 / (t - self._last)
            self.fps = inst if self.fps == 0 else self.fps + self.alpha * (inst - self.fps)
        self._last = t
        self.proc_ms = proc_ms if self.proc_ms == 0 else self.proc_ms + self.alpha * (proc_ms - self.proc_ms)


class VisionService:
    def __init__(self, cfg: ServiceConfig):
        self.cfg = cfg
        self._stop = False

    def stop(self, *_args) -> None:
        self._stop = True

    def run(self) -> None:
        cfg = self.cfg
        signal.signal(signal.SIGINT, self.stop)
        backend = create_backend(cfg.backend, **cfg.backend_kwargs)
        pipeline = VisionPipeline(backend, cfg.pipeline)
        overlay = None
        if cfg.show_debug:
            from .debug_overlay import DebugOverlay
            overlay = DebugOverlay(mirror=cfg.mirror_view, min_visibility=cfg.pipeline.normalization.min_visibility)

        meter = _RateMeter()
        frames = 0
        last_stats = time.perf_counter()
        log.info("streaming %s packets to udp://%s:%d", cfg.packet_format, cfg.host, cfg.port)
        with CameraStream(cfg.camera) as camera, UdpSender(cfg.host, cfg.port) as sender:
            try:
                while not self._stop:
                    frame = camera.read()
                    if frame is None:
                        if camera.is_file:
                            log.info("end of video")
                            break
                        log.warning("no frame from camera")
                        continue
                    t_start = time.perf_counter()
                    state = pipeline.process(frame.image, frame.t, frame.wall_time)
                    pkt = state.packet
                    pkt.latency_ms = (time.perf_counter() - frame.grab_time) * 1000.0
                    sender.send(pkt.encode(cfg.packet_format))
                    meter.tick(frame.t, (time.perf_counter() - t_start) * 1000.0)

                    for e in state.events:
                        log.info("event %-12s foot=%-5s intensity=%.2f", e.action.value, e.foot, e.intensity)

                    if overlay is not None:
                        stats = {"fps": meter.fps, "proc_ms": meter.proc_ms,
                                 "filter": cfg.pipeline.smoothing.method,
                                 "udp": f"{cfg.host}:{cfg.port} {sender.sent}/{sender.dropped}"}
                        cv2.imshow("Vision Service - lower body", overlay.draw(frame.image, state, pipeline.recognizer, stats))
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27):
                            break
                        if key == ord("f"):
                            i = _FILTER_CYCLE.index(cfg.pipeline.smoothing.method)
                            pipeline.set_smoothing(_FILTER_CYCLE[(i + 1) % len(_FILTER_CYCLE)])
                            log.info("smoothing -> %s", cfg.pipeline.smoothing.method)
                        elif key == ord("r"):
                            pipeline.reset()
                            log.info("pipeline reset (floor / scale re-learned)")
                        elif key == ord("m"):
                            overlay.mirror = not overlay.mirror

                    frames += 1
                    now = time.perf_counter()
                    if now - last_stats >= cfg.stats_interval_s:
                        log.info("fps=%.1f proc=%.1fms e2e=%.1fms sent=%d dropped=%d send=%.0fus backend=%s",
                                 meter.fps, meter.proc_ms, pkt.latency_ms, sender.sent, sender.dropped,
                                 sender.last_send_us, pkt.backend)
                        last_stats = now
                    if cfg.max_frames is not None and frames >= cfg.max_frames:
                        break
            finally:
                backend.close()
                if overlay is not None:
                    cv2.destroyAllWindows()
