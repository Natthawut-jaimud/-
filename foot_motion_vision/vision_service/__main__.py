"""CLI entry point: ``python -m vision_service --help``."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

from .capture import CameraConfig
from .pipeline import PipelineConfig
from .pose import HybridConfig
from .service import ServiceConfig, VisionService
from .transport import DEFAULT_HOST, DEFAULT_PORT


def _apply_overrides(obj, values: dict, section: str) -> None:
    names = {f.name for f in dataclasses.fields(obj)}
    for key, value in values.items():
        if key not in names:
            raise SystemExit(f"unknown option {section}.{key}; valid: {sorted(names)}")
        setattr(obj, key, value)


def build_config(argv: list[str] | None = None) -> ServiceConfig:
    p = argparse.ArgumentParser(prog="vision_service", description=__doc__)
    src = p.add_argument_group("input")
    src.add_argument("--camera", default="0", help="camera index or path to a video file (default 0)")
    src.add_argument("--width", type=int, default=640)
    src.add_argument("--height", type=int, default=480)
    src.add_argument("--fps", type=int, default=60)
    src.add_argument("--no-mjpg", action="store_true", help="do not request MJPG from the webcam")
    src.add_argument("--exposure", type=float, default=None, help="manual exposure (driver units) to cut motion blur")

    model = p.add_argument_group("pose model")
    model.add_argument("--backend", choices=("mediapipe", "yolo", "hybrid"), default="mediapipe")
    model.add_argument("--variant", choices=("lite", "full", "heavy"), default="full", help="MediaPipe model size")
    model.add_argument("--yolo-model", default="yolov8n-pose.pt")
    model.add_argument("--device", default=None, help="YOLO device, e.g. cpu, cuda:0, mps")

    proc = p.add_argument_group("processing")
    proc.add_argument("--filter", choices=("one_euro", "ema", "none"), default="one_euro")
    proc.add_argument("--min-cutoff", type=float, default=None, help="One Euro min_cutoff (Hz)")
    proc.add_argument("--beta", type=float, default=None, help="One Euro beta")
    proc.add_argument("--ema-alpha", type=float, default=None)
    proc.add_argument("--config", type=Path, default=None,
                      help="JSON with optional sections: smoothing, normalization, gestures, hybrid")
    proc.add_argument("--include-joints", action="store_true", help="add all 10 lower-body joints to JSON packets")

    out = p.add_argument_group("output")
    out.add_argument("--host", default=DEFAULT_HOST)
    out.add_argument("--port", type=int, default=DEFAULT_PORT)
    out.add_argument("--format", choices=("json", "binary"), default="json")
    out.add_argument("--no-debug", action="store_true", help="headless: no OpenCV window")
    out.add_argument("--no-mirror", action="store_true", help="debug view not mirrored")
    out.add_argument("--max-frames", type=int, default=None)
    out.add_argument("--log-level", default="INFO")
    a = p.parse_args(argv)

    logging.basicConfig(level=a.log_level.upper(), format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    pipeline = PipelineConfig(include_joints=a.include_joints)
    pipeline.smoothing.method = a.filter
    if a.min_cutoff is not None:
        pipeline.smoothing.min_cutoff = a.min_cutoff
    if a.beta is not None:
        pipeline.smoothing.beta = a.beta
    if a.ema_alpha is not None:
        pipeline.smoothing.ema_alpha = a.ema_alpha

    hybrid = HybridConfig()
    if a.config is not None:
        data = json.loads(a.config.read_text(encoding="utf-8"))
        targets = {"smoothing": pipeline.smoothing, "normalization": pipeline.normalization,
                   "gestures": pipeline.gestures, "hybrid": hybrid}
        for section, values in data.items():
            if section not in targets:
                raise SystemExit(f"unknown config section {section!r}; valid: {sorted(targets)}")
            _apply_overrides(targets[section], values, section)

    camera = CameraConfig(
        source=int(a.camera) if a.camera.isdigit() else a.camera,
        width=a.width, height=a.height, fps=a.fps, use_mjpg=not a.no_mjpg, exposure=a.exposure,
    )
    return ServiceConfig(
        camera=camera,
        pipeline=pipeline,
        backend=a.backend,
        backend_kwargs={"variant": a.variant, "yolo_model": a.yolo_model, "device": a.device,
                        "hybrid_config": hybrid},
        host=a.host,
        port=a.port,
        packet_format=a.format,
        show_debug=not a.no_debug,
        mirror_view=not a.no_mirror,
        max_frames=a.max_frames,
    )


def main(argv: list[str] | None = None) -> None:
    VisionService(build_config(argv)).run()


if __name__ == "__main__":
    main()
