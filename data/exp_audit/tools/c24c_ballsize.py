"""Cycle 24c: depth from ball size to pick between overlapping boxes.

For each overlap candidate: px/m at that robot (box bottom-centre back-projected to the floor, scale at
z = 1 m) -> expected single-ball blob area pi*(0.075*ppm)^2. Observed: median blob area of the launch track
(scan.pkl tracks). Offset c = median log(observed / expected) on UNambiguous box-linked launches. Rule:
pick the candidate minimising |log(obs / exp_i) - c|; 'combo' uses it only when the tie-break score margin
is < 0.3, else keeps the tie-break pick. Accuracy on the scout-judged cases (c24_truth_cases.json).
"""
import gc
import json
import pickle
from collections import defaultdict

import numpy as np

from common import AUDIT, DIRS, P, load
from autoscout import views

T = json.load(open(AUDIT / "c24_truth_cases.json"))
by_k = defaultdict(list)
for a in T:
    by_k[a["key"]].append(a)


def ppm_at(cam, u, v):
    xy = cam.backproject(np.array([[u, v]]), z=0.0)[0]
    if not np.all(np.isfinite(xy)):
        return np.nan
    p = cam.project(np.array([[xy[0], xy[1], 1.0], [xy[0] + 0.5, xy[1], 1.0], [xy[0], xy[1] + 0.5, 1.0]]))
    return float(max(np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[0])) * 2)


logs_unamb = []
cases = []
for k, items in by_k.items():
    r = load(k, "v2")
    rows = json.loads((DIRS["v2"] / k / "dets.json").read_text())
    with open(DIRS["v2"] / k / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    vl = {v.name: v for v in views.PRODUCTIONS[md.prod]}
    idx = {}
    for vname, tracks in md.tracks_all.items():
        for tr in tracks:
            if len(tr.pts) >= 4:
                idx[(vname, round(tr.pts[0][0], 3), round(tr.pts[0][1], 1), round(tr.pts[0][2], 1))] = tr
    cams = md.cams
    del md
    gc.collect()
    by_vt = defaultdict(lambda: defaultdict(list))
    for d in rows:
        by_vt[d[1]][d[0]].append(d)
    tk = {v: np.array(sorted(x)) for v, x in by_vt.items()}
    amb_i = {a["i"] for a in items}

    def box_of(s, team):
        ts = tk[s["view"]]; j = int(np.argmin(np.abs(ts - s["t"])))
        ds = [d for d in by_vt[s["view"]][ts[j]] if d[7] == team]
        return ds[0] if ds else None

    def obs_exp(s, team):
        tr = idx.get((s["view"], round(s["t"], 3), round(s["u"], 1), round(s["v"], 1)))
        cam_t = cams.get(s["view"])
        d = box_of(s, team) if s["view"] in tk else None
        if tr is None or cam_t is None or d is None:
            return None
        ox, oy = vl[s["view"]].offset(1920, 1080)
        ppm = ppm_at(cam_t.at(s["t"]), (d[2] + d[4]) / 2 - ox, d[5] - oy)
        if not np.isfinite(ppm) or ppm <= 0:
            return None
        area = float(np.median([q[3] for q in tr.pts]))
        return np.log(max(area, 1.0) / (np.pi * (0.075 * ppm) ** 2))

    # calibration on a sample of unambiguous box launches of this match
    for si, s in enumerate(r["shots"]):
        if s["how"] == "box" and s["team"] is not None and si not in amb_i and si % 3 == 0:
            x = obs_exp(s, s["team"])
            if x is not None:
                logs_unamb.append(x)
    for a in items:
        s = r["shots"][a["i"]]
        lr = {t: obs_exp(s, int(t)) for t in a["cands"]}
        cases.append((a, lr))
    print(k, flush=True)
c = float(np.median(logs_unamb))
print(f"\nunambiguous launches: median log(observed / expected ball area) {c:.2f} (n={len(logs_unamb)}), IQR "
      f"{np.quantile(logs_unamb, .25):.2f}..{np.quantile(logs_unamb, .75):.2f}")
res = defaultdict(list)
for a, lr in cases:
    ok_lr = {t: v for t, v in lr.items() if v is not None}
    o = sorted(a["cands"], key=lambda t: a["cands"][t]["score"])
    margin = a["cands"][o[1]]["score"] - a["cands"][o[0]]["score"]
    size_pick = min(ok_lr, key=lambda t: abs(ok_lr[t] - c)) if len(ok_lr) == len(lr) else str(a["pick"])
    combo = size_pick if margin < 0.3 else str(a["pick"])
    key = "same" if not a["cross"] else "cross"
    for rule, pk in (("tie", str(a["pick"])), ("ballsize", size_pick), ("combo", combo)):
        res[(rule, key)].append(pk == str(a["truth"]))
        res[(rule, "all")].append(pk == str(a["truth"]))
for rule in ("tie", "ballsize", "combo"):
    print(f"   {rule:9s} all {100 * np.mean(res[(rule, 'all')]):5.1f} %  same-alliance {100 * np.mean(res[(rule, 'same')]):5.1f} % "
          f"(n={len(res[(rule, 'same')])})  cross {100 * np.mean(res[(rule, 'cross')]):5.1f} %")
