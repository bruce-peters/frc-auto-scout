"""Cross-view error of a recovered end camera by time distance to its nearest solved pose: does
interpolating across long gaps without tags hurt? Read-only.
Usage: probe_gaps.py SCAN_DIR MATCH:VIEW [MATCH:VIEW ...]
"""
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(r"C:\Users\bruce\Documents\Coding\auto-scout")
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from autoscout import pipeline as P, robots, views  # noqa: E402

src = Path(sys.argv[1])
for item in sys.argv[2:]:
    key, vname = item.split(":")
    with open(src / key / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    cam, main = md.cams[vname], md.cams["main"]
    vl = {v.name: v for v in views.PRODUCTIONS[md.prod]}
    ts = np.asarray(cam.times)

    def place(d, c, limit):
        ox, oy = vl[d.view].offset(1920, 1080)
        c = c.at(d.t)
        xy = c.backproject(np.array([[(d.box[0] + d.box[2]) / 2 - ox, d.box[3] - oy]]), z=0.03)[0]
        if not np.all(np.isfinite(xy)):
            return None
        dv = xy - c.center_world()[:2]
        rng = float(np.hypot(*dv))
        xy = xy + P.ROBOT_HALF_DEPTH * dv / max(rng, 1e-6)
        return xy if robots.in_field(xy) and (limit is None or rng <= limit) else None

    by_t = {}
    for d in md.dets:
        if d.view in ("main", vname):
            by_t.setdefault(round(d.t, 3), []).append(d)
    rows = []
    for t, ds in by_t.items():
        mains = [(d.alliance, p) for d in ds if d.view == "main" and (p := place(d, main, None)) is not None]
        for d in ds:
            if d.view != vname:
                continue
            p = place(d, cam, P.END_VIEW_RANGE_COV)
            q = [np.hypot(*(p - pm)) for al, pm in mains if al == d.alliance] if p is not None else []
            if q:
                rows.append((float(np.min(np.abs(ts - d.t))), min(q)))
    r = np.array(rows)
    print(f"{key} {vname}: {len(ts)} poses, largest gap {np.diff(ts).max():.0f} s")
    for lo, hi in ((0, 2), (2, 6), (6, 12), (12, 1e9)):
        m = (r[:, 0] >= lo) & (r[:, 0] < hi)
        if m.any():
            print(f"   {lo:2.0f}-{hi if hi < 1e9 else 'inf'} s from a pose: cross-view median {np.median(r[m, 1]):.2f} m "
                  f"q75 {np.quantile(r[m, 1], .75):.2f} (n {m.sum()})")
    del md
