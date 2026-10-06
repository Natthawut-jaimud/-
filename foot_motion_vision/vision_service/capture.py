"""Low-latency camera capture.

A background thread reads the camera as fast as it delivers frames and keeps
only the newest one.  The processing loop therefore always works on the most
recent image instead of draining an OpenCV buffer of stale frames, which is
the most common source of "hidden" latency in webcam pipelines.
"""

from __future__ import annotations

import logging
import platform
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    source: int | str = 0          # camera index or video file path
    width: int = 640
    height: int = 480
    fps: int = 60
    use_mjpg: bool = True          # most USB webcams only reach 60 FPS in MJPG
    buffer_size: int = 1
    auto_exposure_off: bool = False  # short fixed exposure reduces motion blur
    exposure: float | None = None


@dataclass
class Frame:
    image: np.ndarray        # BGR
    t: float                 # media time, seconds (perf_counter for cameras, frame clock for files)
    wall_time: float         # time.time() at grab (sent to the client)
    index: int
    grab_time: float = 0.0   # time.perf_counter() when the frame was grabbed (latency reference)


def _default_backend() -> int:
    system = platform.system()
    if system == "Windows":
        return cv2.CAP_DSHOW  # MSMF is slow to open and often caps at 30 FPS
    if system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_V4L2


class CameraStream:
    """Threaded "latest frame wins" camera reader.

    Video files are read synchronously (every frame, no dropping) so offline
    tests are deterministic; timestamps then come from the file's frame rate.
    """

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.is_file = isinstance(cfg.source, str) and not cfg.source.isdigit()
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Condition()
        self._latest: Frame | None = None
        self._consumed_index = -1
        self._running = False
        self._thread: threading.Thread | None = None
        self._index = 0
        self._file_fps = 30.0
        self._t0 = 0.0
        self.actual_fps = 0.0
        self.actual_size = (cfg.width, cfg.height)

    # ----------------------------------------------------------- lifecycle
    def open(self) -> "CameraStream":
        cfg = self.cfg
        if self.is_file:
            cap = cv2.VideoCapture(str(cfg.source))
        else:
            index = int(cfg.source)
            cap = cv2.VideoCapture(index, _default_backend())
            if not cap.isOpened():
                cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video source {cfg.source!r}")

        if not self.is_file:
            if cfg.use_mjpg:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
            cap.set(cv2.CAP_PROP_FPS, cfg.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, cfg.buffer_size)
            if cfg.auto_exposure_off:
                # 0.25 = manual on V4L2, 0 on DirectShow; harmless if unsupported.
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25 if platform.system() == "Linux" else 0)
            if cfg.exposure is not None:
                cap.set(cv2.CAP_PROP_EXPOSURE, cfg.exposure)
        else:
            self._file_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        self.actual_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        log.info("video source %r opened at %dx%d @ %.1f FPS (requested %dx%d @ %d)",
                 cfg.source, *self.actual_size, self.actual_fps, cfg.width, cfg.height, cfg.fps)
        self._cap = cap
        self._t0 = time.perf_counter()
        if not self.is_file:
            self._running = True
            self._thread = threading.Thread(target=self._reader, name="camera-reader", daemon=True)
            self._thread.start()
        return self

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "CameraStream":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------- reading
    def _reader(self) -> None:
        assert self._cap is not None
        failures = 0
        while self._running:
            ok, img = self._cap.read()
            t = time.perf_counter()
            if not ok:
                failures += 1
                if failures > 100:
                    log.error("camera stopped delivering frames")
                    self._running = False
                    with self._lock:
                        self._lock.notify_all()
                    return
                time.sleep(0.005)
                continue
            failures = 0
            with self._lock:
                self._latest = Frame(img, t, time.time(), self._index, t)
                self._index += 1
                self._lock.notify_all()

    def read(self, timeout: float = 1.0) -> Frame | None:
        """Return the newest frame not returned before (None on timeout/end)."""
        if self.is_file:
            assert self._cap is not None
            ok, img = self._cap.read()
            if not ok:
                return None
            t = self._t0 + self._index / self._file_fps
            frame = Frame(img, t, time.time(), self._index, time.perf_counter())
            self._index += 1
            return frame
        deadline = time.perf_counter() + timeout
        with self._lock:
            while self._running and (self._latest is None or self._latest.index == self._consumed_index):
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return None
                self._lock.wait(remaining)
            if self._latest is None or self._latest.index == self._consumed_index:
                return None
            self._consumed_index = self._latest.index
            return self._latest
