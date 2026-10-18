"""Cycle 22: cancmp red_end vs main disagreement (2-4 m in ~9 matches). Which view is wrong, and does it cost red?

Per cancmp match (output/ scans): red-alliance boxes seen in red_end and main at the same detection time;
nearest-pair offset (red_end - main) in field metres (median dx, dy), and per view the share of red-side
floor points that are physically impossible (beyond the red wall x > L - 0.3, or inside the red hub
footprint |dx|,|dy| < 0.55 m of the hub centre). Plus the red vs blue share error vs OPR in matches with
and without the disagreement, and how often red tracks split (segments per red team).
"""
import gc
import pickle
from collections import defaultdict

import numpy as np

from common import DIRS, F, P, load, match_keys, opr, tba
from autoscout import views

BAD = {"2026cancmp_qm5", "2026cancmp_qm90", "2026cancmp_qm99", "2026cancmp_sf10m1", "2026cancmp_sf8m1",
       "2026cancmp_qm52", "2026cancmp_qm114", "2026cancmp_qm80", "2026cancmp_qm70"}
HUB = F.hub_center("red")


def xy_of(cams, vl, d):
    cam = cams.get(d.view)
    if cam is None:
        return None
    c = cam.at(d.t)
    ox, oy = vl[d.view].offset(1920, 1080)
    p = c.backproject(np.array([[(d.box[0] + d.box[2]) / 2 - ox, d.box[3] - oy]]), z=0.03)[0]
    if not np.all(np.isfinite(p)):
        return None
    ray = p - c.center_world()[:2]
    return p + P.ROBOT_HALF_DEPTH * ray / max(np.hypot(*ray), 1e-6)


rows = []
for k in [k for k in match_keys("v2") if k.startswith("2026cancmp")]:
    with open(DIRS["v2"] / k / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    vl = {v.name: v for v in views.PRODUCTIONS[md.prod]}
    by_t = defaultdict(list)
    pts = {"main": [], "red_end": []}
    for d in md.dets:
        if d.alliance == "red" and d.view in ("main", "red_end"):
            p = xy_of(md.cams, vl, d)
            if p is not None:
                by_t[round(d.t, 2)].append((d.view, p))
                if p[0] > F.L / 2:
                    pts[d.view].append(p)
    off = []
    for t, ds in by_t.items():
        a = [p for v, p in ds if v == "red_end"]; b = [p for v, p in ds if v == "main"]
        for p in a:
            if b:
                q = min(b, key=lambda q: np.hypot(*(p - q)))
                if np.hypot(*(p - q)) < 6:
                    off.append(p - q)
    off = np.array(off).reshape(-1, 2)
    viol = {}
    for v, P_ in pts.items():
        P_ = np.array(P_).reshape(-1, 2)
        if len(P_):
            wall = np.mean(P_[:, 0] > F.L - 0.3)
            hub = np.mean((np.abs(P_[:, 0] - HUB[0]) < 0.55) & (np.abs(P_[:, 1] - HUB[1]) < 0.55))
            viol[v] = (round(float(wall), 3), round(float(hub), 3), len(P_))
    del md
    gc.collect()
    r = load(k, "v2"); info = tba.match(k); op = opr(info.event_key)
    err = {}
    for al in ("red", "blue"):
        tm = info.teams(al)
        f = np.array([r["teams"][str(t)]["fuel"] for t in tm]); o = np.array([op.get(t, np.nan) for t in tm])
        err[al] = round(float(np.abs(f / f.sum() - o / o.sum()).sum() / 2), 3) if f.sum() > 0 and np.all(np.isfinite(o)) else None
    nseg = {al: len(r["identity"][al]) for al in ("red", "blue")}
    rows.append((k, k in BAD, np.round(np.median(off, axis=0), 2).tolist() if len(off) else None, len(off), viol, err, nseg))
    print(rows[-1], flush=True)
for flag in (True, False):
    g = [x for x in rows if x[1] == flag]
    er = [x[5]["red"] for x in g if x[5]["red"] is not None]; eb = [x[5]["blue"] for x in g if x[5]["blue"] is not None]
    print(f"{'disagreeing' if flag else 'agreeing':11s} matches {len(g):2d}: share err vs OPR red {np.mean(er):.3f} blue {np.mean(eb):.3f}; "
          f"segments red {np.mean([x[6]['red'] for x in g]):.0f} blue {np.mean([x[6]['blue'] for x in g]):.0f}")
