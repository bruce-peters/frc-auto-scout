"""Cycle 7c: what would make the auto split better than uniform?

Auto increments only (period_of(match_time(t - 2)) == 'auto'), re-split from result.json signals with:
  base        : split_increments as is
  launch_only : HIDDEN = 0, PRIOR = 0.01 (launch evidence only)
  no_hidden   : HIDDEN = 0
  blend_x     : x * base + (1 - x) * uniform
  no_zone     : zone gating off in auto (z = 1)
Scored vs auto-OPR shares (c6b) and the effect on total-fuel share corr vs fuel OPR.
Usage: c6c_auto_fix.py [v2]
"""
import sys

import numpy as np

from c6b_auto_opr import period_opr
from common import A, TL, load, match_keys, opr, resplit, share_metrics, tba, team_signals

src = sys.argv[1] if len(sys.argv) > 1 else "v2"
R, INFO, SIG = {}, {}, {}
for k in match_keys(src):
    R[k] = load(k, src); INFO[k] = tba.match(k); SIG[k] = team_signals(R[k], INFO[k])
OPA = {}


def auto_split(k, al, mode):
    r, info = R[k], INFO[k]
    tl = TL(r)
    if not tl.counts.get(al):
        return None
    inc = [(t, n) for t, n in A.increments_from_counts(tl.counts[al])
           if tl.match_time(t - 2.0) is not None and A.period_of(tl.match_time(t - 2.0)) == "auto"]
    sig = [SIG[k][t] for t in info.teams(al)]
    H0, P0 = A.HIDDEN, A.PRIOR
    zf = A.TeamSignals.zone_frac
    if mode == "launch_only":
        A.HIDDEN, A.PRIOR = 0.0, 0.01
    elif mode == "no_hidden":
        A.HIDDEN = 0.0
    elif mode == "no_zone":
        A.TeamSignals.zone_frac = lambda self, a, b: 1.0
    try:
        sp = A.split_increments(inc, sig, tl.match_time)
    finally:
        A.HIDDEN, A.PRIOR = H0, P0
        A.TeamSignals.zone_frac = zf
    tot = sum(n for _, n in inc)
    tba_auto = info.hub(al).get("autoCount") or tot
    scale = tba_auto / tot if tot else 1.0
    return np.array([sum(sp.get(t, {}).values()) * scale for t in info.teams(al)])


modes = ["base", "launch_only", "no_hidden", "no_zone"]
res = {m: ([], []) for m in modes + [f"blend_{x}" for x in (0.0, 0.25, 0.5, 0.75)]}
tot_fuel = {m: {} for m in res}
for k in R:
    info = INFO[k]
    ev = info.event_key
    if ev not in OPA:
        OPA[ev] = period_opr(ev, lambda h: h.get("autoCount", 0))
    full = resplit(R[k], info, signals=SIG[k])
    for m in res:
        tot_fuel[m][k] = dict(full)
    for al in ("red", "blue"):
        tm = info.teams(al)
        o = np.array([OPA[ev].get(t, np.nan) for t in tm])
        vals = {}
        for m in modes:
            vals[m] = auto_split(k, al, m)
        if vals["base"] is None:
            continue
        b = vals["base"]
        n = b.sum()
        for x in (0.0, 0.25, 0.5, 0.75):
            vals[f"blend_{x}"] = x * b + (1 - x) * n / 3
        for m, v in vals.items():
            # replace the base auto fuel in the total with this variant's auto fuel
            for t, a_new, a_old in zip(tm, v, b):
                tot_fuel[m][k][t] = full[t] - a_old + a_new
            if n >= 15 and np.all(np.isfinite(o)) and o.sum() > 0:
                res[m][0].extend(v / v.sum()); res[m][1].extend(o / o.sum())
print(f"{len(R)} matches ({src}); auto split vs auto-OPR shares, and total-fuel share metrics vs fuel OPR:")
for m, (v, o) in res.items():
    v, o = np.array(v), np.array(o)
    sm = share_metrics(tot_fuel[m])
    print(f"   {m:12s} auto corr {np.corrcoef(v, o)[0, 1]:+.3f}  auto MAE share {np.mean(np.abs(v - o)):.3f}   | total: corr "
          f"{sm['corr']:.3f} top {sm['top']}/{sm['alliances']} MAE {sm['mae']:.1f}")
