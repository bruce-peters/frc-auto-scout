"""Full frames with every credited launch arc of an analysis result drawn up to that moment, coloured by the
robot it was credited to, plus team-labelled robot boxes (dets.json of output/, read only).

  python data/exp_track/tools/render_arcs.py 2026cc_qm24 70.5 71.5 72.5 73.5 --result data/exp_track/ab/<arm>/2026cc_qm24/result.json --out DIR
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from autoscout import config, views  # noqa: E402
from autoscout.video import Video, video_path  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("match")
ap.add_argument("times", nargs="+", type=float)
ap.add_argument("--result", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tail", type=float, default=0.8, help="draw arcs launched up to this many s before the frame")
ap.add_argument("--label", default="")
a = ap.parse_args()

res = json.loads(Path(a.result).read_text())
dets = json.loads((config.OUTPUT / a.match / "dets.json").read_text())
v = Video(video_path(a.match))
prod = views.detect_production(v.frame_at(a.times[0]))
offs = {vw.name: vw.offset(v.width, v.height) for vw in views.PRODUCTIONS[prod]}
teams = sorted(int(t) for t in res["teams"] if not t.startswith("_"))
pal = [(0, 220, 255), (255, 80, 255), (80, 255, 80), (0, 140, 255), (255, 255, 0), (160, 80, 255)]
col = {t: pal[i % len(pal)] for i, t in enumerate(teams)}
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
tset = np.array(sorted({d[0] for d in dets}))
for t in a.times:
    f = v.frame_at(t).copy()
    td = tset[np.argmin(np.abs(tset - t))]
    for d in dets:
        if d[0] != td or d[7] is None:
            continue
        x0, y0, x1, y1 = (int(c) for c in d[2:6])
        c = col.get(d[7], (200, 200, 200))
        cv2.rectangle(f, (x0, y0), (x1, y1), c, 2)
        cv2.putText(f, str(d[7]), (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    n = 0
    for s in res["shots"]:
        if s["team"] is None or not (t - a.tail <= s["t"] <= t):
            continue
        ox, oy = offs.get(s["view"], (0, 0))
        p = np.array([(q[1] + ox, q[2] + oy) for q in s["pts"] if q[0] <= t + 1e-6])
        if len(p) < 2:
            continue
        c = col.get(s["team"], (200, 200, 200))
        cv2.polylines(f, [p.astype(np.int32).reshape(-1, 1, 2)], False, c, 2)
        cv2.circle(f, tuple(p[0].astype(int)), 4, c, -1)
        n += 1
    cv2.putText(f, f"{a.match}  t={t:.2f}s  {a.label}  credited arcs in the last {a.tail:.1f}s: {n}", (20, 1060),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(str(out / f"{a.match}_{t:.2f}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 88])
print("wrote", len(a.times), "frames to", out)
