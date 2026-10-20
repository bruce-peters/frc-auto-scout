"""Is a 60 fps match video really 60 pictures a second, or 30 pictures each shown twice?

Most FIRST-production webcasts are 30p content in a 60 fps file (57 of 62 here); every Chezy one (69) and 5
Championship (Johnson) matches are true 60p. Tracking FUEL on every frame of a doubled video feeds the tracker
each ball position twice: the motion goes still, jump, still, jump, the split-at-jumps step cuts the tracks
into 2-position pieces and the launch gate drops them (2026cancmp_qm5: 1,068 credited launches at 30 Hz -> 142
at 60 Hz). For those videos 30 Hz (fuel_step 2) is already one analysed frame per picture.

Two tests, both separate the two kinds cleanly on all 131 matches:
* duplicated_pictures(log): from a 60 Hz blob log alone (no video): share of a frame's main-view blobs that
  reappear within 1.5 px in the next frame, for each frame phase; doubled videos 0.40-0.52, true 60p 0.12-0.19.
* duplicated_frames(video): from the video: share of consecutive frames whose difference is tiny next to the
  frames around it; doubled ~0.5 (0.47-0.53), true 60p <= 0.07.
Suggested use (production, not wired here): tracks60.load returns None when duplicated_pictures(log), so the
match keeps its 30 Hz tracks; a 60 Hz scan (Options.fuel_step = 1) of a new match uses fuel_step 2 when
duplicated_frames(video).

  python data/exp_track/tools/dup60.py [key ...]     # prints both tests per match
"""
from __future__ import annotations

import gzip
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from autoscout import config, tracks60  # noqa: E402
from autoscout.video import Video, video_path  # noqa: E402

LOG_REPEAT_MAX = 0.30  # blob-repeat share above this: doubled pictures
FRAME_DUP_MIN = 0.30   # tiny-difference share above this: doubled pictures


def log_repeat(log: dict, view: str = "main") -> float:
    """Higher of the two frame phases' share of blobs repeated (< 1.5 px) from the previous frame."""
    rows = log["blob_log"][view]
    rep = []
    for a, b in zip(rows[:-1], rows[1:]):
        A, B = a[1][:, :2], b[1][:, :2]
        if len(A) and len(B):
            rep.append(float((np.sqrt(((B[:, None] - A[None]) ** 2).sum(-1)).min(1) < 1.5).mean()))
        else:
            rep.append(np.nan)
    rep = np.array(rep)
    return float(max(np.nanmean(rep[0::2]), np.nanmean(rep[1::2])))


def duplicated_pictures(log: dict) -> bool:
    return log.get("fuel_step", 1) == 1 and log_repeat(log) > LOG_REPEAT_MAX


def frame_dup_share(video: Video, t0s=(40, 80, 120, 160), n: int = 121) -> float:
    shares = []
    for t0 in t0s:
        start = int(t0 * video.fps)
        if start + n >= video.n_frames:
            continue
        prev, d = None, []
        for _, f in video.iter_frames(start, start + n):
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[: int(0.65 * f.shape[0])].astype(np.int16)
            if prev is not None:
                d.append(float(np.abs(g - prev).mean()))
            prev = g
        d = np.array(d)
        tiny = [d[i] < 0.35 * np.median(np.sort(d[max(0, i - 15):i + 16])[len(d[max(0, i - 15):i + 16]) // 2:])
                for i in range(len(d))]
        shares.append(float(np.mean(tiny)))
    return float(np.median(shares)) if shares else 0.0


def duplicated_frames(video: Video) -> bool:
    return frame_dup_share(video) > FRAME_DUP_MIN


if __name__ == "__main__":
    for key in sys.argv[1:]:
        p = tracks60.log_path(key)
        lr = log_repeat(pickle.load(gzip.open(p, "rb"))) if p else float("nan")
        fs = frame_dup_share(Video(video_path(key)))
        print(f"{key}: log repeat {lr:.2f} -> {'doubled' if lr > LOG_REPEAT_MAX else 'true 60p'} | "
              f"frame dup share {fs:.2f} -> {'doubled' if fs > FRAME_DUP_MIN else 'true 60p'}")
