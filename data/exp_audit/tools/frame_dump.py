"""Dump annotated frames: robot boxes (dets.json, team labels) and launch points (result.json shots within
+-0.5 s, as crosses with the credited team) at the given video times.

  .venv/Scripts/python data/exp_audit/tools/frame_dump.py 2026cc_qm24 12.36 73.9 --out DIR [--alliance blue]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from autoscout import config, views  # noqa: E402
from autoscout.video import Video, video_path  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("match")
ap.add_argument("times", nargs="+", type=float)
ap.add_argument("--out", required=True)
ap.add_argument("--alliance", default=None)
a = ap.parse_args()

od = config.OUTPUT / a.match
res = json.loads((od / "result.json").read_text())
dets = json.loads((od / "dets.json").read_text())
v = Video(video_path(a.match))
prod = views.detect_production(v.frame_at(a.times[0]))
offs = {vw.name: vw.offset(v.width, v.height) for vw in views.PRODUCTIONS[prod]}
pal = [(0, 255, 255), (255, 0, 255), (0, 255, 0), (0, 128, 255), (255, 255, 0), (128, 0, 255)]
teams = sorted(int(t) for t in res["teams"] if not t.startswith("_"))
col = {t: pal[i % len(pal)] for i, t in enumerate(teams)}
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
tset = sorted({d[0] for d in dets})
for t in a.times:
    f = v.frame_at(t).copy()
    td = min(tset, key=lambda x: abs(x - t))
    for d in dets:
        if d[0] != td or (a.alliance and d[6] != a.alliance):
            continue
        x1, y1, x2, y2 = map(int, d[2:6])
        c = col.get(d[7], (200, 200, 200))
        cv2.rectangle(f, (x1, y1), (x2, y2), c, 2)
        cv2.putText(f, f"{d[7]} s{d[8]}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    for s in res["shots"]:
        if abs(s["t"] - t) > 0.5 or (a.alliance and s["alliance"] != a.alliance):
            continue
        ox, oy = offs.get(s["view"], (0, 0))[:2]
        p = (int(ox + s["u"]), int(oy + s["v"]))
        c = col.get(s["team"], (255, 255, 255))
        cv2.drawMarker(f, p, c, cv2.MARKER_CROSS, 22, 3)
        cv2.putText(f, f"{s['team']} {s['how']} {s['t'] - t:+.2f}", (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    cv2.putText(f, f"{a.match} t={t:.2f} (dets {td:.2f})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    p = out / f"{a.match}_{t:07.2f}.jpg"
    cv2.imwrite(str(p), f, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(p)
