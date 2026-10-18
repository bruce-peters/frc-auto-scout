"""Cycle 11c: per-period attribution quality vs period-specific OPR (TBA hubScore per period).

For each period (auto, transition, shift1-4, endgame): ridge OPR on that period's count over the
event's quals, then within-alliance share corr / MAE of video fuel_by_period vs that OPR, next to a
uniform split. Periods where the alliance's hub is inactive (count 0) are skipped.
"""
import numpy as np

from c6b_auto_opr import period_opr
from common import load, match_keys, tba

PER = {"auto": "autoCount", "transition": "transitionCount", "shift1": "shift1Count", "shift2": "shift2Count",
       "shift3": "shift3Count", "shift4": "shift4Count", "endgame": "endgameCount"}
OP = {}
res = {p: ([], [], []) for p in PER}
for k in match_keys("v2"):
    r = load(k, "v2"); info = tba.match(k); ev = info.event_key
    if ev not in OP:
        OP[ev] = {p: period_opr(ev, lambda h, f=f: h.get(f, 0)) for p, f in PER.items()}
    for al in ("red", "blue"):
        tm = info.teams(al)
        for p, f in PER.items():
            n = info.hub(al).get(f, 0)
            v = np.array([r["teams"][str(t)]["fuel_by_period"].get(p, 0.0) for t in tm])
            o = np.array([OP[ev][p].get(t, np.nan) for t in tm])
            if n < 15 or v.sum() <= 0 or not np.all(np.isfinite(o)) or o.sum() <= 0:
                continue
            res[p][0].extend(v / v.sum()); res[p][1].extend(o / o.sum()); res[p][2].append(n)
print(f"{len(match_keys('v2'))} matches; per period: alliances, share corr vs period-OPR, MAE video vs uniform, mean balls")
for p, (v, o, n) in res.items():
    v, o = np.array(v), np.array(o)
    print(f"   {p:10s} {len(n):4d}  corr {np.corrcoef(v, o)[0, 1]:+.3f}  MAE {np.mean(np.abs(v - o)):.3f} vs uniform "
          f"{np.mean(np.abs(1 / 3 - o)):.3f}  balls {np.mean(n):.0f}")
