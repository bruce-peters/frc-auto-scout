"""Cycle 4: defenders credited with the opponent's launches.

Credited toward-hub launches whose linked robot stands in the OPPONENT's half (>= 1.5 m past the centre
line... i.e. well outside its own zone). For box links: is there an opponent box at the launch point
(same expanded-box test as pipeline.analyse) whose team is inside its own zone? Re-linking those to
the opponent (as toward-hub launches) -> share metrics.
Usage: c4_defense.py [v2|v1|cc_v2]
"""
import copy
import json
import sys
from collections import Counter

import numpy as np

from common import AUDIT, DIRS, F, P, TL, load, match_keys, path_video_times, resplit, scout_metrics, share_metrics, tba
from autoscout import views

src = sys.argv[1] if len(sys.argv) > 1 else "v2"
ZL = {"blue": F.ALLIANCE_ZONE_DEPTH, "red": F.L - F.ALLIANCE_ZONE_DEPTH}


def depth(al, x):
    return (ZL["blue"] - x) if al == "blue" else (x - ZL["red"])


def launch_point(s):
    p = np.array([q[1:] for q in s["pts"][:4]]); t = np.array([q[0] for q in s["pts"][:4]])
    A_ = np.vstack([t - t[0], np.ones(len(t))]).T
    vu = np.linalg.lstsq(A_, p[:, 0], rcond=None)[0][0]; vv = np.linalg.lstsq(A_, p[:, 1], rcond=None)[0][0]
    return s["u"] - vu * 2 / 60, s["v"] - vv * 2 / 60


def box_dist(box, ox, oy, lu, lv):
    x0, y0, x1, y1 = box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy
    bw, bh = x1 - x0, y1 - y0
    ex0, ex1, ey0 = x0 - 0.25 * bw, x1 + 0.25 * bw, y0 - 0.9 * bh
    dx = max(ex0 - lu, 0, lu - ex1); dy = max(ey0 - lv, 0, lv - y1)
    return np.hypot(dx, dy) / max(bh, 1)


cat = Counter()
R, INFO, NEW = {}, {}, {}
examples = []
for k in match_keys(src):
    r = load(k, src); info = tba.match(k); tl = TL(r)
    R[k], INFO[k] = r, info
    vl = {v.name: v for v in views.PRODUCTIONS["chezy" if r["timeline"]["layout"] == "chezy" else "first"]}
    dets = json.loads((DIRS[src] / k / "dets.json").read_text())
    by_vt = {}
    for d in dets:
        by_vt.setdefault(d[1], {}).setdefault(d[0], []).append(d)
    keys_vt = {v: np.array(sorted(x)) for v, x in by_vt.items()}
    paths = {t: path_video_times(r["paths"].get(str(t), []), tl) for t in info.red + info.blue}
    pos = lambda t, at: (P._track_position(paths[t], at) if len(paths[t]) else None)
    r2 = copy.deepcopy(r)
    for s, s2 in zip(r["shots"], r2["shots"]):
        if s["team"] is None or not s["toward"]:
            continue
        al = "red" if s["team"] in info.red else "blue"
        p = pos(s["team"], s["t"])
        if p is None or depth(al, p[0]) > -(F.L / 2 - F.ALLIANCE_ZONE_DEPTH) - 1.5:  # not in the opponent's half
            continue
        cat["linked robot in opponent half"] += 1
        if s["how"] != "box":
            cat[f"  via {s['how']}"] += 1
            continue
        tk = keys_vt.get(s["view"])
        if tk is None or not len(tk):
            continue
        j = int(np.argmin(np.abs(tk - s["t"])))
        ox, oy = vl[s["view"]].offset(1920, 1080)
        lu, lv = launch_point(s)
        opp = "blue" if al == "red" else "red"
        cands = []
        for d in by_vt[s["view"]][tk[j]]:
            if d[6] != opp or d[7] is None:
                continue
            dd = box_dist(d[2:6], ox, oy, lu, lv)
            q = pos(d[7], s["t"])
            if dd <= 0.6 and q is not None and depth(opp, q[0]) >= -0.6:
                cands.append((dd, d[7]))
        if cands:
            cat["  via box, opponent box at launch point (in its zone)"] += 1
            tm = min(cands)[1]
            s2.update(team=tm, alliance=opp, toward=True, conf=s["conf"])
            if len(examples) < 40:
                examples.append({"key": k, "t": s["t"], "mt": tl.match_time(s["t"]), "view": s["view"], "from": s["team"],
                                 "to": tm, "u": s["u"], "v": s["v"]})
        else:
            cat["  via box, no opponent box at launch point"] += 1
    NEW[k] = r2
n_all = sum(1 for k in R for s in R[k]["shots"] if s["team"] is not None and s["toward"])
print(f"{len(R)} matches ({src}); credited toward-hub launches {n_all}")
for c, v in cat.items():
    print(f"   {c:58s} {v:5d}  {100 * v / n_all:5.1f} %")
base = {k: resplit(R[k], INFO[k]) for k in R}
alt = {k: resplit(NEW[k], INFO[k]) for k in R}
print("base   ", share_metrics(base), scout_metrics(base))
print("relink ", share_metrics(alt), scout_metrics(alt))
json.dump(examples, open(AUDIT / f"c4_examples_{src}.json", "w"), indent=1, default=float)
