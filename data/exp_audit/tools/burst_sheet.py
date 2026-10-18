"""Contact sheet of a time window: frames every STEP s with robot boxes (+team), every flight track alive
at that time (is_shot -> yellow, credited launch -> green, other tracks -> grey) drawn on the frame.
Usage: burst_sheet.py SRC MATCH VT0 VT1 [STEP] [VIEW]   (video seconds). Writes data/exp_audit/img/.
"""
import json
import pickle
import sys

import cv2
import numpy as np

from common import AUDIT, DIRS, config, P, load
from autoscout import fuel, views

src, key, vt0, vt1 = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
step = float(sys.argv[5]) if len(sys.argv) > 5 else 0.2
only_view = sys.argv[6] if len(sys.argv) > 6 else None
r = load(key, src)
dets = json.loads((DIRS[src] / key / "dets.json").read_text())
with open(DIRS[src] / key / "scan.pkl", "rb") as fh:
    md = pickle.load(fh)
vl = {v.name: v for v in views.PRODUCTIONS[md.prod]}
tracks = {vn: [tr for tr in trs if tr.pts[0][0] <= vt1 + 0.5 and tr.pts[-1][0] >= vt0 - 0.5] for vn, trs in md.tracks_all.items()}
credited = {(s["view"], round(s["t"], 3), round(s["u"], 1), round(s["v"], 1)): s["team"] for s in r["shots"] if s["team"] is not None}
del md
cap = cv2.VideoCapture(str(config.VIDEOS / f"{key}.mp4"))
W, H = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
sx, sy = W / 1920, H / 1080
det_t = np.array([d[0] for d in dets])
panels = []
for t in np.arange(vt0, vt1, step):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, fr = cap.read()
    if not ok:
        continue
    j = int(np.argmin(np.abs(det_t - t)))
    for d in (d for d in dets if abs(d[0] - det_t[j]) < 1e-6):
        x0, y0, x1, y1 = [int(round(c)) for c in (d[2] * sx, d[3] * sy, d[4] * sx, d[5] * sy)]
        col = (60, 70, 240) if d[6] == "red" else (230, 160, 90)
        cv2.rectangle(fr, (x0, y0), (x1, y1), col, 2)
        cv2.putText(fr, str(d[7]), (x0, max(14, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(fr, str(d[7]), (x0, max(14, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    n_alive = 0
    for vn, trs in tracks.items():
        ox, oy = vl[vn].offset(1920, 1080)
        for tr in trs:
            ts_ = np.array([p[0] for p in tr.pts])
            if not (t - 0.6 <= ts_[0] <= t + 0.02):  # arcs born in the last 0.6 s only
                continue
            n_alive += 1
            pts = np.array([[(p[1] + ox) * sx, (p[2] + oy) * sy] for p in tr.pts if p[0] <= t + 0.02], np.int32)
            if len(pts) < 1:
                continue
            ck = (vn, round(tr.pts[0][0], 3), round(tr.pts[0][1], 1), round(tr.pts[0][2], 1))
            cr = ck in credited
            col = (0, 255, 0) if cr else ((0, 230, 255) if fuel.is_shot(tr, 80.0) else (150, 150, 150))
            if cr:
                cv2.putText(fr, str(credited[ck]), tuple(int(c) + 6 for c in pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.polylines(fr, [pts], False, col, 2)
            cv2.circle(fr, tuple(int(c) for c in pts[0]), 4, col, -1)
    mt = (20 + t - r["timeline"]["teleop_start"]) if t >= r["timeline"]["teleop_start"] else t - r["timeline"]["auto_start"]
    cv2.rectangle(fr, (0, 0), (int(W), 34), (0, 0, 0), -1)
    cv2.putText(fr, f"{key} vt {t:.2f} mt {mt:.1f}  tracks {n_alive}", (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    if only_view:
        x0, y0, x1, y1 = vl[only_view].scaled(int(W), int(H))
        fr = fr[y0:y1, x0:x1]
    panels.append(cv2.resize(fr, (960, int(960 * fr.shape[0] / fr.shape[1]))))
cap.release()
out = AUDIT / "img"
out.mkdir(exist_ok=True)
for i in range(0, len(panels), 6):
    g = panels[i:i + 6]
    g += [np.zeros_like(panels[0])] * (6 - len(g))
    sheet = np.vstack([np.hstack(g[k:k + 2]) for k in range(0, 6, 2)])
    p = out / f"{key}_{vt0:.1f}_{only_view or 'all'}_{i // 6}.jpg"
    cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(p)
