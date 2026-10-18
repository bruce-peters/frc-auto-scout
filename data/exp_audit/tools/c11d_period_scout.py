"""Cycle 11d: per-period shares vs the Chezy scouting (time-valid scout rows).

Scouted shots -> launch time (auto_start + t - 0.8) -> the period its ball would be counted in
(period_of(match_time(launch + LAG_PEAK - 2.0)), mirroring split_increments' period of an increment).
Within-alliance share corr of video fuel_by_period vs scouted shots per period (alliances with all three
robots scouted and >= 10 scouted balls in that period). Also: inactive-hub periods (TBA count 0): scouted
balls shot then (they cannot score) vs video credit (should be 0).
"""
from collections import defaultdict

import numpy as np

from common import A, TL, load, match_keys, tba, time_valid
from autoscout import scouting

bym = scouting.by_match(scouting.load())
res = defaultdict(lambda: ([], []))
inactive_scouted = defaultdict(float)
active_scouted = defaultdict(float)
for k in match_keys("v2"):
    if k not in bym:
        continue
    r = load(k, "v2"); info = tba.match(k); tl = TL(r)
    for al in ("red", "blue"):
        tm = info.teams(al)
        rows = {t: [x for x in bym[k].get(t, []) if time_valid(x)] for t in tm}
        if not all(rows.values()):
            continue
        sc = {t: defaultdict(float) for t in tm}
        for t in tm:
            for x in rows[t]:
                for st in tl.auto_start + x.shots - 0.8:
                    mt = tl.match_time(st + A.LAG_PEAK - 2.0)
                    if mt is not None:
                        sc[t][A.period_of(mt)] += 1.0 / len(rows[t])
        hub = info.hub(al)
        for p in ("auto", "transition", "shift1", "shift2", "shift3", "shift4", "endgame"):
            s = np.array([sc[t][p] for t in tm]); v = np.array([r["teams"][str(t)]["fuel_by_period"].get(p, 0.0) for t in tm])
            if hub.get(f"{p}Count", 0) == 0:
                inactive_scouted[p] += s.sum()
                continue
            active_scouted[p] += s.sum()
            if s.sum() >= 10 and v.sum() > 0:
                res[p][0].extend(s / s.sum()); res[p][1].extend(v / v.sum())
print("per period: share corr video vs scouting (alliances), scouted balls while that hub was inactive")
for p in ("auto", "transition", "shift1", "shift2", "shift3", "shift4", "endgame"):
    s, v = np.array(res[p][0]), np.array(res[p][1])
    c = np.corrcoef(s, v)[0, 1] if len(s) >= 6 else float("nan")
    print(f"   {p:10s} corr {c:+.3f} (n={len(s) // 3:3d})  MAE video {np.mean(np.abs(s - v)) if len(s) else float('nan'):.3f} "
          f"vs uniform {np.mean(np.abs(s - 1 / 3)) if len(s) else float('nan'):.3f}   scouted balls: active {active_scouted[p]:5.0f} "
          f"inactive {inactive_scouted[p]:5.0f}")
