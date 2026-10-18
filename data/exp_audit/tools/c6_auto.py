"""Cycle 6: auto-period credit.

Per robot in auto (TBA autoCount per alliance, split by fuel_by_period['auto']): share within the
alliance vs the scouted auto shots (time-valid scout rows, shots before 20 s), next to a uniform 1/3
split and the alliance's credit sources in auto (launch / prior / hidden). Also: auto launches detected
per robot, and robots that the scouts say did not shoot in auto but got credit.
Usage: c6_auto.py [v2|cc_v2]
"""
import sys
from collections import defaultdict

import numpy as np

from common import A, AUDIT, DIRS, TL, load, match_keys, scout_counts, tba, team_signals, time_valid
from autoscout import scouting

src = sys.argv[1] if len(sys.argv) > 1 else "cc_v2"
bym = scouting.by_match(scouting.load())
# ---- credit sources in auto, all matches
dec = np.zeros(4)
for k in match_keys(src):
    r = load(k, src); info = tba.match(k); tl = TL(r)
    sig = team_signals(r, info)
    for al in ("red", "blue"):
        if not tl.counts.get(al):
            continue
        for t, n in A.increments_from_counts(tl.counts[al]):
            mt = tl.match_time(t - 2.0)
            if mt is None or A.period_of(mt) != "auto":
                continue
            p = []
            for team in info.teams(al):
                ts = sig[team]
                z = ts.zone_frac(t - A.LAG_HI, t - A.LAG_LO)
                s = float((A.kernel(t - ts.shots) * ts.shot_w).sum()) if len(ts.shots) else 0.0
                vis = ts.vis_frac(t - A.LAG_HI, t - A.LAG_LO)
                p.append((z * s, z * A.PRIOR * vis, z * A.HIDDEN * (1 - vis)))
            p = np.array(p)
            if p.sum() <= 1e-9:
                dec[3] += n
            else:
                dec[:3] += n * p.sum(0) / p.sum()
print(f"{len(match_keys(src))} matches ({src}); auto balls by credit source: launch {100 * dec[0] / dec.sum():.1f} % "
      f"prior {100 * dec[1] / dec.sum():.1f} % hidden {100 * dec[2] / dec.sum():.1f} % even {100 * dec[3] / dec.sum():.1f} %")

# ---- vs scouting
rows = []
for key, teams in sorted(bym.items()):
    if not (DIRS[src] / key / "result.json").exists():
        continue
    r = load(key, src); info = tba.match(key)
    for al in ("red", "blue"):
        tm = info.teams(al)
        sc = []
        for t in tm:
            rs = [x for x in teams.get(t, []) if time_valid(x)]
            sc.append(np.mean([(x.shots < 20.5).sum() for x in rs]) if rs else np.nan)
        if not np.all(np.isfinite(sc)):
            continue
        va = np.array([r["teams"][str(t)]["fuel_by_period"].get("auto", 0.0) for t in tm])
        nl = [sum(1 for s in r["shots"] if s["team"] == t and s["toward"] and (TL(r).match_time(s["t"]) or 99) < 20) for t in tm]
        auto_tba = info.hub(al).get("autoCount", 0)
        rows.append((key, al, tm, np.array(sc), va, nl, auto_tba))
sa, sv, su = [], [], []
print(f"\nalliances with all 3 robots scouted (time-valid): {len(rows)}")
print(f"   {'match':14s} {'al':4s} teams              scout-auto        video-auto        launches   TBA auto")
for key, al, tm, sc, va, nl, at in rows:
    if sc.sum() <= 0 or va.sum() <= 0:
        continue
    sa += (sc / sc.sum()).tolist(); sv += (va / va.sum()).tolist(); su += [1 / 3] * 3
    print(f"   {key:14s} {al:4s} {str(tm):18s} {str(np.round(sc).astype(int).tolist()):17s} "
          f"{str(np.round(va).astype(int).tolist()):17s} {str(nl):10s} {at}")
sa, sv = np.array(sa), np.array(sv)
print(f"auto share vs scouting: corr {np.corrcoef(sa, sv)[0, 1]:.3f}, MAE share video {np.mean(np.abs(sa - sv)):.3f} "
      f"vs uniform {np.mean(np.abs(sa - 1 / 3)):.3f}  (n={len(sa)})")
