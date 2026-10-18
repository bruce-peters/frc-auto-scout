"""Cycle 10b: weight PRIOR / HIDDEN by each robot's own launch history instead of flat.

When an increment has little launch evidence, split_increments falls back to PRIOR (visible) and HIDDEN
(hidden) for every in-zone robot alike, so an alliance whose scorer's launches are rarely seen splits
nearly evenly (2026cc_qm12 blue: scouted 126 / 5 / 5, video 82 / 72 / 65). Variant:
    w_i = z_i * (s_i + (PRIOR * vis_i + HIDDEN * (1 - vis_i)) * f_i ** gamma)
    f_i = (1 + H_i) / mean_j (1 + H_j),  H_i = conf-weighted in-zone launches of robot i within +-W s
Mirrors the current split_increments otherwise (AUTO_ZONE_GATE etc.). Metrics vs fuel OPR and scouting.
Usage: c10b_history_prior.py [v2]
"""
import json
import sys

import numpy as np

from common import A, AUDIT, TL, load, match_keys, resplit, scout_metrics, share_metrics, tba, team_signals

src = sys.argv[1] if len(sys.argv) > 1 else "v2"
R, INFO, SIG = {}, {}, {}
for k in match_keys(src):
    R[k] = load(k, src); INFO[k] = tba.match(k); SIG[k] = team_signals(R[k], INFO[k])
ORIG = A.split_increments


def zone_at(ts, t):
    m = (ts.zone_t >= t - 0.5) & (ts.zone_t <= t + 0.5)
    v = ts.zone_v[m]; v = v[~np.isnan(v)]
    return float(v.mean()) if len(v) else 0.5


def make(W, gamma, inzone=True):
    def split(increments, teams, to_match_time, log=None):
        out = {ts.team: {} for ts in teams}
        hz = {}
        for ts in teams:
            if len(ts.shots):
                zl = np.array([zone_at(ts, t) for t in ts.shots]) if inzone else np.ones(len(ts.shots))
                hz[ts.team] = (ts.shots, ts.shot_w * (zl >= 0.5))
            else:
                hz[ts.team] = (np.zeros(0), np.zeros(0))
        for t, n in increments:
            if n <= 0:
                continue
            mt = to_match_time(t - 2.0)
            per = A.period_of(mt) if mt is not None else "unknown"
            gate = A.AUTO_ZONE_GATE or per != "auto"
            H = np.array([hz[ts.team][1][np.abs(hz[ts.team][0] - (t - A.LAG_PEAK)) <= W].sum() for ts in teams])
            f = (1.0 + H) / np.mean(1.0 + H)
            w = np.zeros(len(teams))
            for i, ts in enumerate(teams):
                z = ts.zone_frac(t - A.LAG_HI, t - A.LAG_LO) if gate else 1.0
                s = float((A.kernel(t - ts.shots) * ts.shot_w).sum()) if len(ts.shots) else 0.0
                vis = ts.vis_frac(t - A.LAG_HI, t - A.LAG_LO)
                w[i] = z * (s + (A.PRIOR * vis + A.HIDDEN * (1.0 - vis)) * f[i] ** gamma)
            if w.sum() <= 1e-9:
                w = np.ones(len(teams))
            w = w / w.sum()
            for i, ts in enumerate(teams):
                out[ts.team][per] = out[ts.team].get(per, 0.0) + n * w[i]
        return out
    return split


base = {k: resplit(R[k], INFO[k], signals=SIG[k]) for k in R}
check = None
A.split_increments = make(15.0, 0.0)
same = {k: resplit(R[k], INFO[k], signals=SIG[k]) for k in R}
A.split_increments = ORIG
d = max(abs(same[k][t] - base[k][t]) for k in R for t in base[k])
print(f"{len(R)} matches; gamma=0 reproduces split_increments within {d:.3f} fuel")
print("base", share_metrics(base), scout_metrics(base))
res = {"base": base}
for W in (10.0, 20.0, 40.0, 1e4):
    for g in (0.5, 1.0, 1.5):
        A.split_increments = make(W, g)
        pm = {k: resplit(R[k], INFO[k], signals=SIG[k]) for k in R}
        A.split_increments = ORIG
        res[f"W{W:g}_g{g}"] = pm
        print(f"W {W:6g} gamma {g}:", share_metrics(pm), scout_metrics(pm), flush=True)
json.dump({"fuel": {n: {k: {str(t): f for t, f in v.items()} for k, v in pm.items()} for n, pm in res.items()}},
          open(AUDIT / "c10b_history_prior.json", "w"))
