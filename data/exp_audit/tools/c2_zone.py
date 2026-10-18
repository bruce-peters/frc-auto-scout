"""Cycle 2: alliance-zone gating of launch evidence (field.in_alliance_zone, TeamSignals.zone_frac).

split_increments multiplies a robot's whole launch evidence by the fraction of the increment's window
[t-5.5, t-0.8] it spent in its zone. Questions:
  a) where is the shooter (x vs its zone line) when a credited launch leaves? how many launches are
     made from 'out of zone' (z=0 at launch) or with unknown position?
  b) how much launch evidence is lost because the robot left the zone after shooting (window z << 1
     while z at launch = 1)?
  c) metric effect of gating each launch by the zone at its own launch time instead
Usage: c2_zone.py [v2|v1]
"""
import json
import sys
from collections import Counter

import numpy as np

from common import A, AUDIT, F, TL, load, match_keys, resplit, share_metrics, tba, team_signals

src = sys.argv[1] if len(sys.argv) > 1 else "v2"
keys = match_keys(src)
R = {k: load(k, src) for k in keys}
INFO = {k: tba.match(k) for k in keys}
SIG = {k: team_signals(R[k], INFO[k]) for k in keys}
ZL = {"blue": F.ALLIANCE_ZONE_DEPTH, "red": F.L - F.ALLIANCE_ZONE_DEPTH}


def zone_at(ts, t, half=0.5):
    m = (ts.zone_t >= t - half) & (ts.zone_t <= t + half)
    v = ts.zone_v[m]
    v = v[~np.isnan(v)]
    return float(v.mean()) if len(v) else np.nan


# ---------- a) where launches leave from
depth = []  # metres inside own zone (positive = inside, from the zone line)
cat = Counter()
from common import path_video_times, P

for k in keys:
    r, info = R[k], INFO[k]
    tl = TL(r)
    for s in r["shots"]:
        if s["team"] is None or not s["toward"]:
            continue
        al = "red" if s["team"] in info.red else "blue"
        pts = path_video_times(r["paths"].get(str(s["team"]), []), tl)
        p = P._track_position(pts, s["t"]) if len(pts) else None
        if p is None:
            cat["position unknown"] += 1
            continue
        x = p[0]
        d = (ZL["blue"] - x) if al == "blue" else (x - ZL["red"])
        depth.append(d)
        if d >= 0:
            cat["inside zone"] += 1
        elif d >= -0.6:
            cat["0-0.6 m outside (counted in by margin)"] += 1
        elif d >= -1.5:
            cat["0.6-1.5 m outside (gated out)"] += 1
        else:
            cat[">1.5 m outside (gated out)"] += 1
n = sum(cat.values())
print(f"{len(keys)} matches ({src}); credited launches toward own hub: {n}")
for c, v in cat.most_common():
    print(f"   {c:42s} {v:6d}  {100 * v / n:5.1f} %")
depth = np.array(depth)
hist, edges = np.histogram(np.clip(depth, -4, 6), bins=np.arange(-4, 6.5, 0.5))
print("   depth into own zone (m) histogram:")
for h, e in zip(hist, edges):
    print(f"     {e:5.1f}..{e + 0.5:4.1f} {'#' * int(60 * h / hist.max())} {h}")


# ---------- b) window zone vs zone at launch, weighted by the kernel evidence each launch contributes
lost = tot = 0.0
lost_by = Counter()
for k in keys:
    r, info = R[k], INFO[k]
    tl = TL(r)
    for al in ("red", "blue"):
        if not tl.counts.get(al):
            continue
        inc = A.increments_from_counts(tl.counts[al])
        for team in info.teams(al):
            ts = SIG[k][team]
            if not len(ts.shots):
                continue
            zl = np.array([zone_at(ts, t) for t in ts.shots])
            for t, n_ in inc:
                kw = A.kernel(t - ts.shots) * ts.shot_w
                if kw.sum() <= 0:
                    continue
                zw = ts.zone_frac(t - A.LAG_HI, t - A.LAG_LO)
                e_launch = float((kw * np.nan_to_num(zl, nan=zw)).sum())
                e_win = zw * kw.sum()
                tot += n_ * e_launch
                lost += n_ * max(0.0, e_launch - e_win)
print(f"\n(b) kernel-weighted launch evidence (ball-weighted): {100 * lost / max(tot, 1e-9):.1f} % of the evidence of "
      "in-zone launches is removed because the window zone fraction < zone at launch")


# ---------- c) metric: per-launch zone gating
def split_launch_zone(increments, teams, to_match_time, log=None):
    out = {ts.team: {} for ts in teams}
    zl = {ts.team: (np.array([zone_at(ts, t) for t in ts.shots]) if len(ts.shots) else np.zeros(0)) for ts in teams}
    for t, n_ in increments:
        if n_ <= 0:
            continue
        w = np.zeros(len(teams))
        for i, ts in enumerate(teams):
            z = ts.zone_frac(t - A.LAG_HI, t - A.LAG_LO)
            if len(ts.shots):
                zz = np.nan_to_num(zl[ts.team], nan=z)
                s = float((A.kernel(t - ts.shots) * ts.shot_w * zz).sum())
            else:
                s = 0.0
            vis = ts.vis_frac(t - A.LAG_HI, t - A.LAG_LO)
            w[i] = s + z * (A.PRIOR * vis + A.HIDDEN * (1.0 - vis))
        if w.sum() <= 1e-9:
            w = np.ones(len(teams))
        w = w / w.sum()
        mt = to_match_time(t - 2.0)
        per = A.period_of(mt) if mt is not None else "unknown"
        for i, ts in enumerate(teams):
            out[ts.team][per] = out[ts.team].get(per, 0.0) + n_ * w[i]
    return out


res = {"src": src, "n": len(keys), "launch_position": dict(cat), "lost_evidence_pct": 100 * lost / max(tot, 1e-9)}
base = {k: resplit(R[k], INFO[k], signals=SIG[k]) for k in keys}
res["base"] = share_metrics(base)
orig = A.split_increments
A.split_increments = split_launch_zone
alt = {k: resplit(R[k], INFO[k], signals=SIG[k]) for k in keys}
A.split_increments = orig
res["per_launch_zone"] = share_metrics(alt)
for m in (0.3, 0.6, 1.0, 1.5, 3.0):
    sg = {k: team_signals(R[k], INFO[k], zone_margin=m) for k in keys}
    res[f"margin_{m}"] = share_metrics({k: resplit(R[k], INFO[k], signals=sg[k]) for k in keys})
print("\n(c) share metrics vs fuel OPR:")
for k_, v in res.items():
    if isinstance(v, dict) and "corr" in v:
        print(f"   {k_:18s} corr {v['corr']:.3f} top {v['top']}/{v['alliances']} MAE {v['mae']:.1f}")
json.dump({"res": res, "base_fuel": {k: {str(t): f for t, f in v.items()} for k, v in base.items()},
           "alt_fuel": {k: {str(t): f for t, f in v.items()} for k, v in alt.items()}},
          open(AUDIT / f"c2_zone_{src}.json", "w"), indent=1)
