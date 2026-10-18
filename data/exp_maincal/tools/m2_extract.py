"""Compact per-match data for the main-camera experiments (so fits do not reload the 70 MB scans).

data/exp_maincal/extract/<match>.pkl: robot boxes (t, view, box, alliance, conf), the stored cameras, the
timeline, and the main-view AprilTag sightings at pipeline.calibrate's sample times (every 4 s).
Usage: m2_extract.py [match ...]   (default: every match in data/exp_maincal/before without an extract)
"""
import gc
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(r"C:\Users\bruce\Documents\Coding\auto-scout")
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from autoscout import config, field as F, pipeline as P, views  # noqa: E402
from autoscout.video import Video  # noqa: E402

EXP = ROOT / "data" / "exp_maincal"
SRC = EXP / "before"
DST = EXP / "extract"
DST.mkdir(exist_ok=True)
keys = sys.argv[1:] or sorted(p.name for p in SRC.iterdir() if (p / "scan.pkl").exists() and not (DST / f"{p.name}.pkl").exists())
for key in keys:
    tic = time.time()
    with open(SRC / key / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    tl = md.timeline
    a0 = tl.auto_start if tl.auto_start is not None else P.FIRST_AUTO_START
    tele = tl.teleop_start or (a0 + P.FIRST_TELEOP_GAP)
    vn = {v: i for i, v in enumerate(("main", "blue_end", "red_end"))}
    dets = np.array([(d.t, vn[d.view], *d.box, d.alliance == "red", d.conf) for d in md.dets], np.float64)
    out = {"key": key, "event": md.info.event_key, "prod": md.prod, "cams": md.cams, "a0": a0, "tele": tele,
           "dets": dets, "teams": {"red": md.info.red, "blue": md.info.blue}}
    del md
    gc.collect()
    times = [float(t) for t in np.arange(a0 - 1, tele + config.TELEOP_LEN + 2, 4.0)]  # as process_match
    video = Video(config.VIDEOS / f"{key}.mp4")
    view = next(v for v in views.PRODUCTIONS[out["prod"]] if v.name == "main")
    masks = view.mask_boxes(video.width, video.height)
    tags, size = [], None
    for t in times:
        f = video.frame_at(t)
        if f is None:
            continue
        c = view.crop(f)
        size = c.shape[1::-1]
        tags.append((t, F._detect_masked(c, masks)))
    video.close()
    out.update(tag_obs=tags, size=size, masks=masks)
    with open(DST / f"{key}.pkl", "wb") as fh:
        pickle.dump(out, fh)
    nt = [len(d) for _, d in tags]
    print(f"{key}: {len(dets)} dets, {len(tags)} frames, tags/frame median {np.median(nt):.0f} "
          f"(ids {sorted({i for _, d in tags for i in d})}), {time.time() - tic:.0f}s", flush=True)
