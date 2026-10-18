"""Cycle 5: visual audit of launches linked by projection (how = proj / proj_hidden).

Samples credited toward-hub launches linked by projecting a tracked robot into the arc's view, renders
the frame ~5 samples after launch (cropped around the launch point) with the arc, every box + team,
and the linked robot's projected top (0.55 m) as a cross. Writes data/exp_audit/img/proj_XX.jpg and
proj_shots.json (fill in the verdicts by eye).
Usage: c5_proj_sheet.py SRC SEED N_PER_MATCH match...
"""
import json
import pickle
import random
import sys

import cv2
import numpy as np

from common import AUDIT, DIRS, P, TL, config, load, path_video_times
from autoscout import views

src, seed, per = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
keys = sys.argv[4:]
rng = random.Random(seed)
panels, meta = [], []
for key in keys:
    r = load(key, src); tl = TL(r)
    dets = json.loads((DIRS[src] / key / "dets.json").read_text())
    with open(DIRS[src] / key / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    cams, prod = md.cams, md.prod
    del md
    vl = {v.name: v for v in views.PRODUCTIONS[prod]}
    cand = [s for s in r["shots"] if s["team"] is not None and s["toward"] and s["how"] in ("proj", "proj_hidden")
            and 0 <= (tl.match_time(s["t"]) or -1) <= 160]
    pick = rng.sample(cand, min(per, len(cand)))
    det_t = np.array([d[0] for d in dets])
    cap = cv2.VideoCapture(str(config.VIDEOS / f"{key}.mp4"))
    W, H = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    sx, sy = W / 1920, H / 1080
    for s in pick:
        k = min(5, len(s["pts"]) - 1)
        cap.set(cv2.CAP_PROP_POS_MSEC, s["pts"][k][0] * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        ox, oy = vl[s["view"]].offset(1920, 1080)
        j = int(np.argmin(np.abs(det_t - s["t"])))
        for d in (d for d in dets if abs(d[0] - det_t[j]) < 1e-6):
            x0, y0, x1, y1 = [int(round(c)) for c in (d[2] * sx, d[3] * sy, d[4] * sx, d[5] * sy)]
            col = (60, 70, 240) if d[6] == "red" else (230, 160, 90)
            cv2.rectangle(fr, (x0, y0), (x1, y1), col, 2)
            cv2.putText(fr, str(d[7]), (x0, max(14, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(fr, str(d[7]), (x0, max(14, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        pts = np.array([[(p[1] + ox) * sx, (p[2] + oy) * sy] for p in s["pts"]], np.int32)
        cv2.polylines(fr, [pts], False, (0, 255, 255), 2)
        cv2.circle(fr, tuple(int(c) for c in pts[0]), 6, (0, 255, 0), 2)
        # linked robot's projected top
        pth = path_video_times(r["paths"][str(s["team"])], tl)
        p = P._track_position(pth, s["t"])
        cam = cams.get(s["view"])
        if p is not None and cam is not None:
            u, v = cam.at(s["t"]).project(np.array([[p[0], p[1], 0.55]]))[0]
            cu, cv_ = int((u + ox) * sx), int((v + oy) * sy)
            cv2.drawMarker(fr, (cu, cv_), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 26, 3)
        lu, lv = pts[0]
        cw, ch = 560, 360
        cx0 = int(np.clip(lu - cw / 2, 0, W - cw)); cy0 = int(np.clip(lv - ch * 0.5, 0, H - ch))
        crop = cv2.resize(fr[cy0:cy0 + ch, cx0:cx0 + cw], (780, 500))
        n = len(meta)
        txt = f"#{n} {key} mt {tl.match_time(s['t']):.1f} -> {s['team']} via {s['how']} conf {s['conf']:.2f} {s['view']}"
        cv2.rectangle(crop, (0, 0), (780, 26), (0, 0, 0), -1)
        cv2.putText(crop, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        panels.append(crop)
        meta.append({"n": n, "key": key, "t": s["t"], "mt": round(tl.match_time(s["t"]), 2), "team": s["team"],
                     "how": s["how"], "conf": s["conf"], "view": s["view"],
                     "xy": None if p is None else [round(p[0], 2), round(p[1], 2)]})
    cap.release()
out = AUDIT / "img"
out.mkdir(exist_ok=True)
for i in range(0, len(panels), 4):
    g = panels[i:i + 4] + [np.zeros_like(panels[0])] * (4 - len(panels[i:i + 4]))
    cv2.imwrite(str(out / f"proj_{i // 4:02d}.jpg"), np.vstack([np.hstack(g[:2]), np.hstack(g[2:])]),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
(AUDIT / "proj_shots.json").write_text(json.dumps(meta, indent=1))
print(len(meta), "proj launches rendered")
