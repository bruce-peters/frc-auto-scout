"""Broadcast production layouts: where each camera view sits inside the video frame.

FIRST's 2026 webcast template stacks a wide "main" camera on top and two hub cameras
below (blue-end on the left, red-end on the right) separated by a stone-arch graphic.
Chezy Champs uses the same idea with a thin divider. Boxes are at 1920x1080.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class View:
    name: str
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 at 1920x1080
    masks: tuple[tuple[int, int, int, int], ...] = ()  # overlay graphics inside the view

    def scaled(self, w: int, h: int) -> tuple[int, int, int, int]:
        sx, sy = w / 1920.0, h / 1080.0
        x0, y0, x1, y1 = self.box
        return int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)

    def crop(self, frame: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.scaled(frame.shape[1], frame.shape[0])
        return frame[y0:y1, x0:x1]

    def offset(self, w: int, h: int) -> tuple[int, int]:
        x0, y0, _, _ = self.scaled(w, h)
        return x0, y0

    def mask_boxes(self, w: int, h: int) -> list[tuple[int, int, int, int]]:
        """Overlay boxes in view-local pixel coordinates."""
        sx, sy = w / 1920.0, h / 1080.0
        vx0, vy0, _, _ = self.box
        out = []
        for x0, y0, x1, y1 in self.masks:
            out.append((int((x0 - vx0) * sx), int((y0 - vy0) * sy),
                        int((x1 - vx0) * sx), int((y1 - vy0) * sy)))
        return out


PRODUCTIONS: dict[str, list[View]] = {
    "first": [
        View("main", (0, 0, 1920, 704),
             masks=((0, 0, 1920, 170),)),  # score bar + corner counters
        View("blue_end", (0, 745, 935, 1080), masks=((395, 1010, 935, 1080),)),
        View("red_end", (985, 745, 1920, 1080), masks=((985, 1010, 1525, 1080),
                                                        (1670, 1015, 1920, 1080))),
    ],
    "chezy": [
        View("main", (0, 0, 1920, 640), masks=((480, 0, 1440, 180),)),
        View("blue_end", (958, 645, 1920, 1080)),
        View("red_end", (0, 645, 955, 1080)),
    ],
}


def detect_production(frame: np.ndarray) -> str:
    """FIRST template has a beige stone band at y~705-744; Chezy a dark divider at y~642."""
    h, w = frame.shape[:2]
    sy = h / 1080.0
    band = frame[int(708 * sy):int(740 * sy), int(100 * w / 1920):int(1800 * w / 1920)]
    b, g, r = band.reshape(-1, 3).mean(0)
    if r > 140 and r > b + 30 and band.reshape(-1, 3).std(0).mean() < 45:
        return "first"
    return "chezy"
