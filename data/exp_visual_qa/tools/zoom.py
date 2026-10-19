"""Zoomed crops (annotated over raw) of one region at several video times, for reading bumpers and balls.

  .venv/Scripts/python data/exp_visual_qa/tools/zoom.py MATCH ALLIANCE x0,y0,x1,y1 T1 T2 ... --out NAME [--w 900]
  (crop in full-frame 1920x1080 px; output data/exp_visual_qa/img/NAME.jpg; --raw_only / --ann_only)
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_burst import IMG, MatchCtx, draw, save_jpg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("match")
ap.add_argument("alliance")
ap.add_argument("crop")
ap.add_argument("times", nargs="+", type=float)
ap.add_argument("--out", required=True)
ap.add_argument("--w", type=int, default=900)
ap.add_argument("--cols", type=int, default=2, help="2 = annotated | raw; 1 = annotated only")
ap.add_argument("--raw_only", action="store_true")
a = ap.parse_args()
ctx = MatchCtx(a.match)
info_teams = ctx.meta["teams_red"] if a.alliance == "red" else ctx.meta["teams_blue"]
x0, y0, x1, y1 = [int(c) for c in a.crop.split(",")]
shots_al = [s for s in ctx.res["shots"] if s["alliance"] == a.alliance]
shots_ot = [s for s in ctx.res["shots"] if s["alliance"] != a.alliance]
rows = []
for t in a.times:
    f = ctx.vid.frame_at(t)
    ann = draw(ctx, f, t, a.alliance, info_teams, shots_al, shots_ot)
    s = a.w / (x1 - x0)
    pa = cv2.resize(ann[y0:y1, x0:x1], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    pr = cv2.resize(f[y0:y1, x0:x1], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    for p in (pa, pr):
        cv2.putText(p, f"t={t:.2f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    if a.raw_only:
        rows.append(pr)
    elif a.cols == 1:
        rows.append(pa)
    else:
        rows.append(np.hstack([pa, np.full((pa.shape[0], 4, 3), 20, np.uint8), pr]))
img = np.vstack(rows)
kb = save_jpg(IMG / f"{a.out}.jpg", img)
print(IMG / f"{a.out}.jpg", kb, "KB", "teams", info_teams)
