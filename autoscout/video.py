"""Video download (yt-dlp) and frame access."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from .config import VIDEOS


def video_path(match_key: str) -> Path:
    return VIDEOS / f"{match_key}.mp4"


def download(match_key: str, youtube_key: str, max_height: int = 1080) -> Path:
    """Download a match video (H.264 video-only, <= max_height). Returns the path."""
    out = video_path(match_key)
    if out.exists() and out.stat().st_size > 1_000_000:
        return out
    yt_key, _, start = youtube_key.partition("?t=")
    fmt = (
        f"bestvideo[height<={max_height}][vcodec^=avc1]/"
        f"bestvideo[height<={max_height}]/best[height<={max_height}]"
    )
    cmd = [sys.executable, "-m", "yt_dlp", "-q", "--no-progress", "-f", fmt,
           "-o", str(out.with_suffix(".%(ext)s")), f"https://www.youtube.com/watch?v={yt_key}"]
    if start:
        # full-stream links ("?t=1215"): grab a 5-minute window starting there
        s = int(start.rstrip("s"))
        cmd[1:1] = []
        cmd += ["--download-sections", f"*{max(0, s - 20)}-{s + 280}"]
    subprocess.run(cmd, check=True)
    if not out.exists():  # yt-dlp may have chosen another container
        cands = list(VIDEOS.glob(f"{match_key}.*"))
        if cands:
            cands[0].rename(out)
    return out


class Video:
    """Thin wrapper over cv2.VideoCapture with time-based access."""

    def __init__(self, path: Path | str):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise IOError(f"cannot open {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.n_frames / self.fps
        self._pos = 0

    def frame_at(self, t: float) -> np.ndarray | None:
        idx = int(round(t * self.fps))
        return self.frame(idx)

    def frame(self, idx: int) -> np.ndarray | None:
        if idx != self._pos:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = self.cap.read()
        self._pos = idx + 1
        return f if ok else None

    def iter_frames(self, start_idx: int = 0, end_idx: int | None = None, step: int = 1):
        """Yield (idx, frame) sequentially; decodes every frame but yields every `step`."""
        end_idx = self.n_frames if end_idx is None else min(end_idx, self.n_frames)
        if start_idx != self._pos:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        idx = start_idx
        while idx < end_idx:
            if step > 1 and (idx - start_idx) % step:
                ok = self.cap.grab()
                f = None
            else:
                ok, f = self.cap.read()
            if not ok:
                break
            if f is not None:
                yield idx, f
            idx += 1
        self._pos = idx

    def close(self):
        self.cap.release()
