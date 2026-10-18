"""Sample full frames from downloaded match videos for pseudo-labeling / review."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autoscout.config import DATA, VIDEOS  # noqa: E402
from autoscout.video import Video  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA / "frame_pool"))
    ap.add_argument("--per_video", type=int, default=40)
    ap.add_argument("--t0", type=float, default=8.0)
    ap.add_argument("--t1", type=float, default=185.0)
    ap.add_argument("--missing_only", action="store_true", help="only videos with no frames in --out yet")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for vp in sorted(VIDEOS.glob("*.mp4")):
        if a.missing_only and any(out.glob(f"{vp.stem}_*.jpg")):
            continue
        v = Video(vp)
        t1 = min(a.t1, v.duration - 2)
        ts = np.sort(rng.uniform(a.t0, t1, a.per_video))
        n = 0
        for t in ts:
            fn = out / f"{vp.stem}_{t:07.2f}.jpg"
            if fn.exists():
                continue
            f = v.frame_at(float(t))
            if f is None:
                continue
            cv2.imwrite(str(fn), f, [cv2.IMWRITE_JPEG_QUALITY, 92])
            n += 1
        v.close()
        print(vp.stem, n, flush=True)


if __name__ == "__main__":
    main()
