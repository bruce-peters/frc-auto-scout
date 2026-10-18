"""Cycle 16d: dead / parked robots collect PRIOR (and HIDDEN) credit. Gate it by recent movement.

activity_i(t) = min(1, D_i / D0), D_i = extent (max pairwise span, m) of robot i's OBSERVED positions in
[t - W, t]; no observed fix in the window -> treated as active (1.0, so hidden robots keep HIDDEN).
Variant split: w_i = z_i * (s_i + (PRIOR * vis_i + HIDDEN * (1 - vis_i)) * activity_i).
Also prints activity stats for the scouted dead/absent robots (comments) vs everyone.
"""
import json

import numpy as np

from common import A, AUDIT, TL, load, match_keys, path_video_times, resplit, scout_metrics, share_metrics, tba, team_signals

R, INFO, SIG, PTS = {}, {}, {}, {}
for k in match_keys("v2"):
    r = load(k, "v2"); info = tba.match(k); tl = TL(r)
    R[k], INFO[k], SIG[k] = r, info, team_signals(r, info)
    PTS[k] = {t: path_video_times(r["paths"].get(str(t), []), tl) for t in info.red + info.blue}
ORIG = A.split_increments


def extent(p, a, b):
    m = (p[:, 0] >= a) & (p[:, 0] <= b)
    if m.sum() < 3:
        return None
    q = p[m, 1:]
    return float(np.hypot(*(q.max(0) - q.min(0))))


def make(k, W, D0):
    def split(increments, teams, to_match_time, log=None):
        out = {ts.team: {} for ts in teams}
        for t, n in increments:
            if n <= 0:
                continue
            mt = to_match_time(t - 2.0)
            per = A.period_of(mt) if mt is not None else "unknown"
            gate = A.AUTO_ZONE_GATE or per != "auto"
            w = np.zeros(len(teams))
            for i, ts in enumerate(teams):
                z = ts.zone_frac(t - A.LAG_HI, t - A.LAG_LO) if gate else 1.0
                s = float((A.kernel(t - ts.shots) * ts.shot_w).sum()) if len(ts.shots) else 0.0
                vis = ts.vis_frac(t - A.LAG_HI, t - A.LAG_LO)
                p = PTS[k].get(ts.team)
                e = extent(p, t - W, t) if p is not None and len(p) else None
                act = 1.0 if e is None else min(1.0, e / D0)
                w[i] = z * (s + (A.PRIOR * vis + A.HIDDEN * (1.0 - vis)) * act)
            if w.sum() <= 1e-9:
                w = np.ones(len(teams))
            w = w / w.sum()
            for i, ts in enumerate(teams):
                out[ts.team][per] = out[ts.team].get(per, 0.0) + n * w[i]
        return out
    return split


DEAD = [("2026cc_qm18", 3045), ("2026cc_qm19", 4499), ("2026cc_qm23", 4499), ("2026cc_qm30", 5507), ("2026cc_qm3", 6036)]
for k, t in DEAD:
    if k in PTS:
        p = PTS[k][t]
        tl = TL(R[k])
        ex = [extent(p, a, a + 20) for a in np.arange(tl.t0(), tl.t1() - 20, 20)]
        print(f"{k} {t}: 20 s movement extents {[None if e is None else round(e, 1) for e in ex]}")
allx = []
for k in R:
    tl = TL(R[k])
    for t, p in PTS[k].items():
        if len(p):
            allx += [e for e in (extent(p, a, a + 20) for a in np.arange(tl.t0(), tl.t1() - 20, 20)) if e is not None]
print(f"all robots, 20 s movement extent quantiles 5/10/25/50: {np.round(np.quantile(allx, [.05, .1, .25, .5]), 2)}")
base = {k: resplit(R[k], INFO[k], signals=SIG[k]) for k in R}
print("base", share_metrics(base), scout_metrics(base), flush=True)
res = {"base": base}
for W, D0 in ((20, 1.0), (20, 2.0), (30, 1.5), (10, 1.0)):
    pm = {}
    for k in R:
        A.split_increments = make(k, W, D0)
        pm[k] = resplit(R[k], INFO[k], signals=SIG[k])
        A.split_increments = ORIG
    res[f"W{W}_D{D0}"] = pm
    print(f"W {W} D0 {D0}:", share_metrics(pm), scout_metrics(pm), flush=True)
json.dump({"fuel": {n: {k: {str(t): f for t, f in v.items()} for k, v in pm.items()} for n, pm in res.items()}},
          open(AUDIT / "c16d_activity.json", "w"))
