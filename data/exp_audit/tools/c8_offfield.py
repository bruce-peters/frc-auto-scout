"""Cycle 8: robot boxes whose floor point back-projects off the field (d.xy None in analyse()).

Per match and view: share of boxes with no field position (same back-projection as analyse(): box
bottom-centre at z 0.03, + ROBOT_HALF_DEPTH along the ray, robots.in_field; end views also > 12 m).
Per 4 s pose window of the main camera: off-field share, and the pose jump to the neighbouring poses
(rotation deg / centre m) -> are a few bad pose fits responsible?
Usage: c8_offfield.py [v2] [match ...]
"""
import gc
import json
import pickle
import sys
from collections import defaultdict

import cv2
import numpy as np

from common import AUDIT, DIRS, P, match_keys, tba
from autoscout import robots, views

src = sys.argv[1] if len(sys.argv) > 1 else "v2"
keys = sys.argv[2:] or match_keys(src)


def floor_xy(cam, d, ox, oy):
    c = cam.at(d.t)
    u, v = (d.box[0] + d.box[2]) / 2 - ox, d.box[3] - oy
    xy = c.backproject(np.array([[u, v]]), z=0.03)[0]
    if not np.all(np.isfinite(xy)):
        return None, None
    ray = xy - c.center_world()[:2]
    rng = float(np.hypot(*ray))
    xy = xy + P.ROBOT_HALF_DEPTH * ray / max(rng, 1e-6)
    return xy, rng


out = {}
ev_tot = defaultdict(lambda: np.zeros(2))
for k in keys:
    with open(DIRS[src] / k / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    vl = {v.name: v for v in views.PRODUCTIONS[md.prod]}
    per_view = defaultdict(lambda: np.zeros(2))
    win = defaultdict(lambda: np.zeros(2))
    for d in md.dets:
        cam = md.cams.get(d.view)
        if cam is None:
            continue
        ox, oy = vl[d.view].offset(1920, 1080)
        xy, rng = floor_xy(cam, d, ox, oy)
        off = xy is None or not robots.in_field(xy) or (d.view != "main" and rng > P.END_VIEW_RANGE_COV)
        per_view[d.view] += (1, off)
        if d.view == "main":
            win[cam.pose_index(d.t)] += (1, off)
    cm = md.cams.get("main")
    jumps = []
    if cm is not None:
        for i in range(len(cm.times)):
            R0 = cv2.Rodrigues(np.array(cm.rvecs[i]))[0]
            nb = [j for j in (i - 1, i + 1) if 0 <= j < len(cm.times)]
            rot = [np.degrees(np.linalg.norm(cv2.Rodrigues(R0.T @ cv2.Rodrigues(np.array(cm.rvecs[j]))[0])[0])) for j in nb]
            cen = [np.linalg.norm(-R0.T @ np.array(cm.tvecs[i]) + cv2.Rodrigues(np.array(cm.rvecs[j]))[0].T @ np.array(cm.tvecs[j]))
                   for j in nb]
            n, o = win.get(i, (0, 0))
            jumps.append((round(cm.times[i], 1), int(n), round(o / max(n, 1), 2), round(min(rot), 2), round(min(cen), 2)))
    ev = tba.match(k).event_key
    ev_tot[ev] += per_view["main"]
    out[k] = {"per_view": {v: [int(a), int(b)] for v, (a, b) in per_view.items()}, "main_windows": jumps,
              "main_rms": cm.rms if cm is not None else None}
    pv = " ".join(f"{v} {b / max(a, 1):.2f}" for v, (a, b) in sorted(per_view.items()))
    bad = [j for j in jumps if j[1] >= 20 and j[2] >= 0.5]
    print(f"{k:18s} off-field share: {pv}; main pose windows >= 50 % off: {len(bad)}/{len(jumps)}", flush=True)
    del md
    gc.collect()
print("\nmain view off-field share by event:")
for e, (a, b) in sorted(ev_tot.items()):
    print(f"   {e:12s} {b / a:.3f}  ({int(a)} boxes)")
json.dump(out, open(AUDIT / f"c8_offfield_{src}.json", "w"), indent=0)
