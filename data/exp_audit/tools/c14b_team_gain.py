"""Cycle 14b: per-team launch gain pooled over the event (balls per detected launch), leave-one-match-out.

Dump shooters (5026, 254, ...) produce fewer separate launch arcs per ball than one-at-a-time shooters,
so split_increments (weight ~ number of detected launches) under-credits them. Fit per event:
  y_b = sum_r g_r * x_rb   (1 s bins of counter increments; x_rb = kernel-spread, zone-gated, conf-weighted
                            launches of robot r in that alliance-match; ridge towards the pooled gain)
over every OTHER match of the event, then split each increment with w_r = z_r * (g_r * s_r + PRIOR*vis + HIDDEN*(1-vis)).
Scored vs the Chezy scouting (and OPR as a sanity check).
"""
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import nnls

from common import A, TL, load, match_keys, resplit, scout_metrics, share_metrics, tba, team_signals

R, INFO, SIG = {}, {}, {}
for k in match_keys("v2"):
    R[k] = load(k, "v2"); INFO[k] = tba.match(k); SIG[k] = team_signals(R[k], INFO[k])
by_ev = defaultdict(list)
for k in R:
    by_ev[INFO[k].event_key].append(k)


def design(k):
    """rows: (y, {team: x}) per 1 s bin of each alliance."""
    r, info = R[k], INFO[k]
    tl = TL(r)
    out = []
    for al in ("red", "blue"):
        if not tl.counts.get(al):
            continue
        inc = A.increments_from_counts(tl.counts[al])
        t0, t1 = tl.t0(), tl.t1() + 5
        edges = np.arange(t0, t1, 1.0)
        c = (edges[:-1] + edges[1:]) / 2
        y = np.zeros(len(c))
        for t, n in inc:
            i = int((t - t0) // 1.0)
            if 0 <= i < len(y):
                y[i] += n
        xs = {}
        for team in info.teams(al):
            ts = SIG[k][team]
            x = np.zeros(len(c))
            for st, sw in zip(ts.shots, ts.shot_w):
                kk = A.kernel(c - st)
                if kk.sum() > 0:
                    mt = tl.match_time(st + A.LAG_PEAK - 2.0)
                    auto = mt is not None and A.period_of(mt) == "auto"
                    z = 1.0 if (auto and not A.AUTO_ZONE_GATE) else ts.zone_frac(st - 0.5, st + 0.5)
                    x += sw * z * kk / kk.sum()
            xs[team] = x
        out.append((y, xs))
    return out


D = {k: design(k) for k in R}


def fit_gains(keys, ridge=3.0):
    teams = sorted({t for k in keys for _, xs in D[k] for t in xs})
    idx = {t: i for i, t in enumerate(teams)}
    Xr, yr = [], []
    for k in keys:
        for y, xs in D[k]:
            X = np.zeros((len(y), len(teams)))
            for t, x in xs.items():
                X[:, idx[t]] = x
            Xr.append(X); yr.append(y)
    X, y = np.vstack(Xr), np.concatenate(yr)
    g0, _ = nnls(X.sum(1, keepdims=True), y)
    lam = np.sqrt(ridge * max(1.0, (X ** 2).sum() / X.shape[1]))
    g, _ = nnls(np.vstack([X, lam * np.eye(len(teams))]), np.concatenate([y, lam * np.full(len(teams), g0[0])]))
    return {t: g[idx[t]] / max(g0[0], 1e-9) for t in teams}  # relative to the pooled gain


ORIG = A.split_increments


def make(gains):
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
                w[i] = z * (gains.get(ts.team, 1.0) * s + A.PRIOR * vis + A.HIDDEN * (1.0 - vis))
            if w.sum() <= 1e-9:
                w = np.ones(len(teams))
            w = w / w.sum()
            for i, ts in enumerate(teams):
                out[ts.team][per] = out[ts.team].get(per, 0.0) + n * w[i]
        return out
    return split


base = {k: resplit(R[k], INFO[k], signals=SIG[k]) for k in R}
print(f"{len(R)} matches; base", share_metrics(base), scout_metrics(base), flush=True)
for ridge in (1.0, 3.0, 10.0):
    pm = {}
    gl = {}
    for ev, ks in by_ev.items():
        for k in ks:
            g = fit_gains([x for x in ks if x != k], ridge)
            gl[k] = g
            A.split_increments = make(g)
            pm[k] = resplit(R[k], INFO[k], signals=SIG[k])
            A.split_increments = ORIG
    print(f"ridge {ridge}:", share_metrics(pm), scout_metrics(pm), flush=True)
    g5026 = [gl[k].get(5026) for k in gl if gl[k].get(5026) is not None and k.startswith("2026cc")]
    print(f"   5026 gain at cc (leave-one-out) median {np.median(g5026):.2f}; 254: "
          f"{np.median([gl[k][254] for k in gl if 254 in gl[k] and k.startswith('2026cc')]):.2f}", flush=True)
