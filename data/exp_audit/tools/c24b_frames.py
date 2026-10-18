"""Frames of same-alliance overlap launches where the pick disagrees with the scouted shooter."""
import json
import random

import cv2
import numpy as np

from common import AUDIT, DIRS, TL, config, load
from autoscout import views

T = json.load(open(AUDIT / "c24_truth_cases.json"))
wrong = [a for a in T if not a["cross"] and str(a["pick"]) != str(a["truth"])]
random.Random(3).shuffle(wrong)
seen, pick = set(), []
for a in wrong:
    if a["key"] not in seen:
        pick.append(a); seen.add(a["key"])
    if len(pick) == 6:
        break
panels = []
for a in pick:
    k = a["key"]; r = load(k, "v2"); tl = TL(r)
    s = r["shots"][a["i"]]
    prod = "chezy" if r["timeline"]["layout"] == "chezy" else "first"
    vl = {v.name: v for v in views.PRODUCTIONS[prod]}
    dets = json.loads((DIRS["v2"] / k / "dets.json").read_text())
    dt = np.array([d[0] for d in dets]); j = int(np.argmin(np.abs(dt - s["t"])))
    cap = cv2.VideoCapture(str(config.VIDEOS / f"{k}.mp4"))
    cap.set(cv2.CAP_PROP_POS_MSEC, s["pts"][min(3, len(s["pts"]) - 1)][0] * 1000)
    ok, fr = cap.read(); cap.release()
    if not ok:
        continue
    ox, oy = vl[s["view"]].offset(1920, 1080)
    for d in (d for d in dets if abs(d[0] - dt[j]) < 1e-6 and d[1] == s["view"]):
        x0, y0, x1, y1 = [int(round(c)) for c in d[2:6]]
        col = (0, 255, 255) if d[7] == a["pick"] else ((0, 255, 0) if d[7] == a["truth"] else (200, 200, 200))
        cv2.rectangle(fr, (x0, y0), (x1, y1), col, 2)
        cv2.putText(fr, str(d[7]), (x0, max(14, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(fr, str(d[7]), (x0, max(14, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    pts = np.array([[p[1] + ox, p[2] + oy] for p in s["pts"]], np.int32)
    cv2.polylines(fr, [pts], False, (255, 0, 255), 2)
    cv2.circle(fr, tuple(int(c) for c in pts[0]), 6, (255, 0, 255), 2)
    lu, lv = pts[0]
    cw, ch = 600, 380
    cx0 = int(np.clip(lu - cw / 2, 0, 1920 - cw)); cy0 = int(np.clip(lv - ch * 0.4, 0, 1080 - ch))
    crop = fr[cy0:cy0 + ch, cx0:cx0 + cw].copy()
    cv2.rectangle(crop, (0, 0), (cw, 24), (0, 0, 0), -1)
    cv2.putText(crop, f"{k} mt {tl.match_time(s['t']):.1f} {s['view']} pick {a['pick']} (yellow) scouted {a['truth']} (green)",
                (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    panels.append(crop)
while len(panels) % 2:
    panels.append(np.zeros_like(panels[0]))
cv2.imwrite(str(AUDIT / "img" / "overlap_wrong.jpg"), np.vstack([np.hstack(panels[i:i + 2]) for i in range(0, len(panels), 2)]),
            [cv2.IMWRITE_JPEG_QUALITY, 85])
print(len(panels))
