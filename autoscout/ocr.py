"""Bumper team-number reading, constrained to the three teams on the robot's alliance.

Open OCR of 10-30 px digits is unreliable, but here every read only has to choose between
three known team numbers (from TBA) — so partial reads ("502", "026", "S026") still carry
evidence. Evidence is accumulated over a whole track segment.
"""
from __future__ import annotations

import cv2
import numpy as np

from .overlay import reader


def edit_distance(a: str, b: str) -> int:
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
            prev = cur
    return dp[-1]


def similarity(read: str, team: str) -> float:
    """1.0 for an exact read; partial credit for substrings / one-off errors."""
    if not read:
        return 0.0
    if read == team:
        return 1.0
    if len(read) >= 2 and read in team:
        return 0.45 + 0.5 * len(read) / len(team) - 0.2
    d = edit_distance(read, team)
    return max(0.0, 1.0 - d / max(len(team), len(read)) - 0.1)


def bumper_crop(frame: np.ndarray, box) -> np.ndarray:
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    h = y1 - y0
    w = x1 - x0
    x0 = max(0, x0 - int(0.05 * w))
    x1 = min(frame.shape[1], x1 + int(0.05 * w))
    yb0 = max(0, y0 + int(0.45 * h))
    yb1 = min(frame.shape[0], y1 + int(0.05 * h))
    return frame[yb0:yb1, x0:x1]


PREP_H = 128  # bumper crops are upscaled to this height before OCR (vs 64: ~35 % more crops name a team)


def prep_bumper(crop: np.ndarray, target_h: int | None = None) -> list[np.ndarray]:
    """White digits on a red/blue bumper -> dark text on light background variants."""
    if crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 12:
        return []
    s = max(1.0, (target_h or PREP_H) / crop.shape[0])
    crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    mn = crop.min(axis=2)  # white -> high, saturated colour -> low
    mn = cv2.normalize(mn, None, 0, 255, cv2.NORM_MINMAX)
    inv = 255 - mn
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(inv)
    return [inv, clahe]


def sharpness(crop: np.ndarray) -> float:
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def read_numbers(crop: np.ndarray) -> list[tuple[str, float]]:
    reads = []
    for img in prep_bumper(crop):
        try:
            res = reader().readtext(img, allowlist="0123456789", detail=1, paragraph=False,
                                    text_threshold=0.5, low_text=0.3)
        except Exception:
            continue
        for _, txt, conf in res:
            txt = txt.strip()
            if 2 <= len(txt) <= 5:
                reads.append((txt, float(conf)))
    return reads


def team_evidence(reads: list[tuple[str, float]], teams: list[int]) -> np.ndarray:
    """Score vector over `teams` from a set of reads (sum of conf * similarity)."""
    ev = np.zeros(len(teams))
    for txt, conf in reads:
        sims = np.array([similarity(txt, str(t)) for t in teams])
        if sims.max() < 0.5:
            continue
        # only the best-matching team gets credit, weighted by how decisive the read is
        best = int(np.argmax(sims))
        margin = sims[best] - np.partition(sims, -2)[-2] if len(sims) > 1 else sims[best]
        if margin <= 0.05:
            continue
        ev[best] += conf * sims[best] * min(1.0, margin * 2)
    return ev
