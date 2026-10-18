"""Cycle 11b: 'visible' for HIDDEN credit = seen by a camera that actually catches this robot's launches.

Scouted-shot recall (c11): robot boxed in its own-end view 62-63 %, main view only 25 %, no box 12 %.
split_increments treats any fused fix as 'visible' (PRIOR only, no HIDDEN credit). Variant: per 0.5 s
grid sample, vis = 1 if a box of the team in its OWN end view within VIS_WINDOW, alpha if boxed only in
other views, 0 if not boxed at all (dets.json team column). alpha = 1 ~ box-based version of today.
Metrics vs fuel OPR (all output/ matches) and scouting.
Usage: c11b_vis_views.py
"""
import json
from collections import defaultdict

import numpy as np

from common import AUDIT, DIRS, P, TL, load, match_keys, resplit, scout_metrics, share_metrics, tba, team_signals

R, INFO, SIG, COV = {}, {}, {}, {}
for k in match_keys("v2"):
    r = load(k, "v2"); info = tba.match(k)
    R[k], INFO[k], SIG[k] = r, info, team_signals(r, info)
    dets = json.loads((DIRS["v2"] / k / "dets.json").read_text())
    own = defaultdict(list); other = defaultdict(list)
    for d in dets:
        if d[7] is None:
            continue
        al = "red" if d[7] in info.red else "blue"
        (own if d[1] == f"{al}_end" else other)[d[7]].append(d[0])
    COV[k] = {t: (np.sort(own.get(t, [])), np.sort(other.get(t, []))) for t in info.red + info.blue}


def near(arr, g, w):
    if not len(arr):
        return np.zeros(len(g), bool)
    j = np.searchsorted(arr, g)
    a = np.abs(arr[np.clip(j - 1, 0, len(arr) - 1)] - g)
    b = np.abs(arr[np.clip(j, 0, len(arr) - 1)] - g)
    return np.minimum(a, b) <= w


base = {k: resplit(R[k], INFO[k], signals=SIG[k]) for k in R}
print(f"{len(R)} matches  base (fused-fix visibility):", share_metrics(base), scout_metrics(base))
out = {"base": base}
for alpha in (1.0, 0.6, 0.3, 0.0):
    pm = {}
    for k in R:
        sig = {}
        for t, ts in SIG[k].items():
            if not len(ts.vis_t):
                sig[t] = ts
                continue
            o, x = COV[k][t]
            g = ts.vis_t
            v = np.where(near(o, g, P.VIS_WINDOW), 1.0, np.where(near(x, g, P.VIS_WINDOW), alpha, 0.0))
            ts2 = type(ts)(ts.team, ts.shots, ts.shot_w, ts.zone_t, ts.zone_v, ts.vis_t, v)
            sig[t] = ts2
        pm[k] = resplit(R[k], INFO[k], signals=sig)
    out[f"alpha{alpha}"] = pm
    print(f"   own-end=1, other views={alpha}:", share_metrics(pm), scout_metrics(pm), flush=True)
json.dump({"fuel": {n: {k: {str(t): f for t, f in v.items()} for k, v in pm.items()} for n, pm in out.items()}},
          open(AUDIT / "c11b_vis_views.json", "w"))
