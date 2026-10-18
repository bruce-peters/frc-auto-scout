"""Cycle 5b: two failure patterns of projection-linked launches, counted over all matches.

A) box fall-through: pipeline.analyse step (1) finds a robot box at the launch point (bd <= 0.6) but that
   box has no team (no field position -> no segment), so step (2) projects EVERY tracked robot of BOTH
   alliances and gives the arc to the nearest one. Counted: proj/proj_hidden credited launches with a
   team-None box within the box-link distance; split by whether that box's alliance matches the team credited.
B) hub-top starts: the arc's first point lies inside the projected airspace over a hub opening
   (field.hub_region_3d(al, 'top'), z 1.25-3.0 m) -> balls bouncing out of / over the hub.
Then: share metrics with (a) cross-alliance fall-throughs dropped, (b) all fall-throughs dropped,
(c) hub-top starts dropped, (d) a + c.
Usage: c5b_proj_stats.py [v2]
"""
import copy
import gc
import json
import pickle
import sys
from collections import Counter

import numpy as np

from common import AUDIT, DIRS, load, match_keys, resplit, scout_metrics, share_metrics, tba
from autoscout import field as F, views

src = sys.argv[1] if len(sys.argv) > 1 else "v2"


def launch_point(s):
    p = np.array([q[1:] for q in s["pts"][:4]]); t = np.array([q[0] for q in s["pts"][:4]])
    A_ = np.vstack([t - t[0], np.ones(len(t))]).T
    vu = np.linalg.lstsq(A_, p[:, 0], rcond=None)[0][0]; vv = np.linalg.lstsq(A_, p[:, 1], rcond=None)[0][0]
    return s["u"] - vu * 2 / 60, s["v"] - vv * 2 / 60


def box_dist(box, ox, oy, lu, lv):
    x0, y0, x1, y1 = box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy
    bw, bh = x1 - x0, y1 - y0
    dx = max(x0 - 0.25 * bw - lu, 0, lu - x1 - 0.25 * bw); dy = max(y0 - 0.9 * bh - lv, 0, lv - y1)
    return np.hypot(dx, dy) / max(bh, 1)


cnt = Counter()
R, INFO, VAR = {}, {}, {v: {} for v in ("drop_cross_fallthrough", "drop_all_fallthrough", "drop_hubtop", "drop_cross+hubtop")}
for k in match_keys(src):
    r = load(k, src); info = tba.match(k)
    R[k], INFO[k] = r, info
    with open(DIRS[src] / k / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    cams, prod = md.cams, md.prod
    del md
    gc.collect()
    vl = {v.name: v for v in views.PRODUCTIONS[prod]}
    dets = json.loads((DIRS[src] / k / "dets.json").read_text())
    by_vt = {}
    for d in dets:
        by_vt.setdefault(d[1], {}).setdefault(d[0], []).append(d)
    tk = {v: np.array(sorted(x)) for v, x in by_vt.items()}
    flags = []
    for s in r["shots"]:
        f = set()
        if s["team"] is not None and s["toward"]:
            cnt[("credited", s["how"])] += 1
            lu, lv = launch_point(s)
            if s["how"] != "box" and s["view"] in tk:
                ts_ = tk[s["view"]]
                j = int(np.argmin(np.abs(ts_ - s["t"])))
                if abs(ts_[j] - s["t"]) <= 0.2:
                    ox, oy = vl[s["view"]].offset(1920, 1080)
                    near = [d for d in by_vt[s["view"]][ts_[j]] if box_dist(d[2:6], ox, oy, lu, lv) <= 0.6]
                    if near:
                        best = min(near, key=lambda d: box_dist(d[2:6], ox, oy, lu, lv))
                        if best[7] is None:
                            al_c = "red" if s["team"] in info.red else "blue"
                            f.add("fallthrough")
                            f.add("cross" if best[6] != al_c else "same")
            cam = cams.get(s["view"])
            if cam is not None:
                c = cam.at(s["t"])
                for al in ("red", "blue"):
                    p = c.project(F.hub_region_3d(al, "top"))
                    import cv2
                    hull = cv2.convexHull(p.astype(np.float32))
                    if cv2.pointPolygonTest(hull, (float(s["u"]), float(s["v"])), False) >= 0:
                        f.add("hubtop")
                        break
            for x in f:
                cnt[(x, s["how"])] += 1
        flags.append(f)
    for name, drop in (("drop_cross_fallthrough", {"cross"}), ("drop_all_fallthrough", {"fallthrough"}),
                       ("drop_hubtop", {"hubtop"}), ("drop_cross+hubtop", {"cross", "hubtop"})):
        r2 = copy.deepcopy(r)
        r2["shots"] = [s for s, f in zip(r["shots"], flags) if not (f & drop)]
        VAR[name][k] = resplit(r2, info)
    print(k, "done", flush=True)
n = sum(v for (a, b), v in cnt.items() if a == "credited")
print(f"\n{len(R)} matches ({src}); credited toward-hub launches {n}")
for (a, b), v in sorted(cnt.items()):
    print(f"   {a:12s} {b:12s} {v:6d}  {100 * v / n:5.1f} % of credited")
base = {k: resplit(R[k], INFO[k]) for k in R}
print("base   ", share_metrics(base), scout_metrics(base))
for name, pm in VAR.items():
    print(f"{name:22s}", share_metrics(pm), scout_metrics(pm))
json.dump({"cnt": {f"{a}|{b}": v for (a, b), v in cnt.items()},
           "fuel": {"base": base, **VAR}}, open(AUDIT / f"c5b_{src}.json", "w"), default=float, indent=1)
