"""Common interface for pose backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..landmarks import PoseResult


class PoseBackend(ABC):
    name: str = "base"

    @abstractmethod
    def process(self, image_bgr: np.ndarray, t: float) -> PoseResult | None:
        """Run inference. ``t`` is the capture time in seconds (monotonic).

        Returns ``None`` when no person is detected.
        """

    def close(self) -> None:  # pragma: no cover - default no-op
        pass

    def __enter__(self) -> "PoseBackend":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
