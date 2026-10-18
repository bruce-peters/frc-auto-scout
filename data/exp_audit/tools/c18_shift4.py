"""Cycle 18: shift 4 / endgame credit vs the Chezy scouting.

Per period (auto .. endgame) and scouted alliance (all three robots scouted with time_valid rows): TV share
error of video fuel_by_period vs scouted shots in that period (ball period = period_of(match_time(launch
+ LAG_PEAK - 2))). Bootstrap of shift4 vs the other teleop periods. Then possible causes:
  * burst sizes per period (stockpile dumps after an inactive shift -> volleys, cycle 17 undercount)
  * credit sources per period (launch / prior / hidden)
  * climbing robots (TBA endGameTowerRobotN != None): their scouted vs credited share in the last 20 s
"""
from collections import defaultdict

import numpy as np

from common import A, TL, load, match_keys, tba, team_signals, time_valid
from autoscout import scouting

PERIODS = ("auto", "transition", "shift1", "shift2", "shift3", "shift4", "endgame")
bym = scouting.by_match(scouting.load())
err = defaultdict(list)            # period -> [(key, TV err)]
bursts = defaultdict(list)         # period -> burst sizes (scouted balls)
src = defaultdict(lambda: np.zeros(3))
climb = defaultdict(lambda: np.zeros(2))  # 'climber'/'non' -> (scouted balls last 20 s, credited balls last 20 s)
for k in match_keys("v2"):
    if not k.startswith("2026cc_qm") or k not in bym:
        continue
    r = load(k, "v2"); info = tba.match(k); tl = TL(r)
    sig = team_signals(r, info)
    for al in ("red", "blue"):
        tm = info.teams(al)
        rows = {t: [x for x in bym[k].get(t, []) if time_valid(x)] for t in tm}
        if not all(rows.values()):
            continue
        hub = info.hub(al)
        sc = {t: defaultdict(float) for t in tm}
        late_sc = {t: 0.0 for t in tm}
        for t in tm:
            for x in rows[t]:
                st = np.sort(tl.auto_start + x.shots - 0.8)
                for s_ in st:
                    mt = tl.match_time(s_ + A.LAG_PEAK - 2.0)
                    if mt is not None:
                        sc[t][A.period_of(mt)] += 1.0 / len(rows[t])
                        if mt >= 140:
                            late_sc[t] += 1.0 / len(rows[t])
                if len(st):
                    for b in np.split(st, np.where(np.diff(st) > 1.5)[0] + 1):
                        mt = tl.match_time(b[0] + A.LAG_PEAK - 2.0)
                        if mt is not None:
                            bursts[A.period_of(mt)].append(len(b) / len(rows[t]) * len(rows[t]))
        for p in PERIODS:
            if hub.get(f"{p}Count", 0) == 0:
                continue
            s = np.array([sc[t][p] for t in tm]); v = np.array([r["teams"][str(t)]["fuel_by_period"].get(p, 0.0) for t in tm])
            if s.sum() >= 10 and v.sum() > 0:
                err[p].append((k, float(np.abs(s / s.sum() - v / v.sum()).sum() / 2)))
        # credit sources per period + late credit per robot
        late_v = {t: 0.0 for t in tm}
        for t, n in A.increments_from_counts(tl.counts.get(al) or []):
            mt = tl.match_time(t - 2.0)
            if mt is None:
                continue
            per = A.period_of(mt)
            p_ = []
            for team in tm:
                ts = sig[team]
                z = ts.zone_frac(t - A.LAG_HI, t - A.LAG_LO) if (A.AUTO_ZONE_GATE or per != "auto") else 1.0
                s_ = float((A.kernel(t - ts.shots) * ts.shot_w).sum()) if len(ts.shots) else 0.0
                vis = ts.vis_frac(t - A.LAG_HI, t - A.LAG_LO)
                p_.append((z * s_, z * A.PRIOR * vis, z * A.HIDDEN * (1 - vis)))
            p_ = np.array(p_)
            if p_.sum() > 0:
                src[per] += n * p_.sum(0) / p_.sum()
                if mt >= 140:
                    w = p_.sum(1) / p_.sum()
                    for team, wi in zip(tm, w):
                        late_v[team] += n * wi
        bd = info.breakdown[al]
        for i, t in enumerate(tm, start=1):
            c = "climber" if bd.get(f"endGameTowerRobot{i}", "None") != "None" else "non"
            climb[c] += (late_sc[t], late_v[t])
print("per period: TV share error vs scouting (alliances), mean scouted burst size, credit sources L/P/H")
for p in PERIODS:
    e = np.array([x[1] for x in err[p]])
    s = src[p] / max(src[p].sum(), 1e-9)
    print(f"   {p:10s} err {e.mean() if len(e) else float('nan'):.3f} (n={len(e):2d})  burst size mean "
          f"{np.mean(bursts[p]) if bursts[p] else float('nan'):5.1f}  share of balls in bursts >= 20: "
          f"{np.sum([b for b in bursts[p] if b >= 20]) / max(np.sum(bursts[p]), 1):.2f}  sources {np.round(s, 2).tolist()}")
s4 = np.array([x[1] for x in err["shift4"]])
oth = np.array([x[1] for p in ("transition", "shift1", "shift2", "shift3") for x in err[p]])
rng = np.random.default_rng(0)
d = [rng.choice(s4, len(s4)).mean() - rng.choice(oth, len(oth)).mean() for _ in range(5000)]
print(f"shift4 - other teleop shifts: {s4.mean() - oth.mean():+.3f}, 95 % CI [{np.quantile(d, .025):+.3f}, {np.quantile(d, .975):+.3f}]")
for c, (a, b) in climb.items():
    print(f"   last 20 s, {c:8s}: scouted balls {a:6.0f}, credited balls {b:6.0f}")
tot_s = sum(v[0] for v in climb.values()); tot_v = sum(v[1] for v in climb.values())
for c, (a, b) in climb.items():
    print(f"   last 20 s share of {c}: scouted {a / max(tot_s, 1e-9):.2f} vs credited {b / max(tot_v, 1e-9):.2f}")
