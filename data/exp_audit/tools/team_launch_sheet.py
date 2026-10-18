"""Frames of one team's credited launches in one match: arc + every box with its assigned team, crop around
the launch point (bumper numbers readable in the end views). Usage: team_launch_sheet.py SRC MATCH TEAM [N]"""
import json
import sys

import cv2
import numpy as np

from common import AUDIT, DIRS, TL, config, load
from autoscout import views

src, key, team = sys.argv[1], sys.argv[2], int(sys.argv[3])
N = int(sys.argv[4]) if len(sys.argv) > 4 else 8
r = load(key, src); tl = TL(r)
prod = "chezy" if r["timeline"]["layout"] == "chezy" else "first"
vl = {v.name: v for v in views.PRODUCTIONS[prod]}
dets = json.loads((DIRS[src] / key / "dets.json").read_text())
det_t = np.array([d[0] for d in dets])
sh = [s for s in r["shots"] if s["team"] == team and s["toward"]]
idx = np.linspace(0, len(sh) - 1, min(N, len(sh))).astype(int) if sh else []
cap = cv2.VideoCapture(str(config.VIDEOS / f"{key}.mp4"))
panels = []
for i in idx:
    s = sh[i]
    k = min(4, len(s["pts"]) - 1)
    cap.set(cv2.CAP_PROP_POS_MSEC, s["pts"][k][0] * 1000)
    ok, fr = cap.read()
    if not ok:
        continue
    ox, oy = vl[s["view"]].offset(1920, 1080)
    j = int(np.argmin(np.abs(det_t - s["t"])))
    for d in (d for d in dets if abs(d[0] - det_t[j]) < 1e-6):
        x0, y0, x1, y1 = [int(round(c)) for c in d[2:6]]
        col = (0, 255, 255) if d[7] == team else ((60, 70, 240) if d[6] == "red" else (230, 160, 90))
        cv2.rectangle(fr, (x0, y0), (x1, y1), col, 2)
        cv2.putText(fr, str(d[7]), (x0, max(14, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(fr, str(d[7]), (x0, max(14, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    pts = np.array([[p[1] + ox, p[2] + oy] for p in s["pts"]], np.int32)
    cv2.polylines(fr, [pts], False, (0, 255, 0), 2)
    cv2.circle(fr, tuple(int(c) for c in pts[0]), 7, (0, 255, 0), 2)
    lu, lv = pts[0]
    cw, ch = 640, 400
    cx0 = int(np.clip(lu - cw / 2, 0, 1920 - cw)); cy0 = int(np.clip(lv - ch * 0.4, 0, 1080 - ch))
    crop = fr[cy0:cy0 + ch, cx0:cx0 + cw].copy()
    cv2.rectangle(crop, (0, 0), (cw, 24), (0, 0, 0), -1)
    cv2.putText(crop, f"{key} mt {tl.match_time(s['t']):.1f} -> {team} via {s['how']} {s['view']} conf {s['conf']:.2f}", (5, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    panels.append(crop)
cap.release()
while len(panels) % 2:
    panels.append(np.zeros_like(panels[0]))
rowsimg = [np.hstack(panels[i:i + 2]) for i in range(0, len(panels), 2)]
out = AUDIT / "img" / f"team_{key}_{team}.jpg"
cv2.imwrite(str(out), np.vstack(rowsimg), [cv2.IMWRITE_JPEG_QUALITY, 85])
print(out, len(panels))
