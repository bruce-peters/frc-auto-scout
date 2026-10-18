"""Cycle 14: team 5026 at Chezy - is it systematically under-credited, and why?

For every scouted cc qual: 5026's video share vs scouted share; recall of its scouted shots (credited
launch within +-1 s) by camera coverage, vs the other robots; where it shoots from; how its credited
launches were linked (view / box / proj); its credit sources; burstiness (balls per scouted second).
"""
import json
from collections import Counter, defaultdict

import numpy as np

from common import A, DIRS, F, P, TL, load, match_keys, path_video_times, scout_counts, tba, team_signals, time_valid
from autoscout import scouting

TEAM = 5026
bym = scouting.by_match(scouting.load())
sc = scout_counts()
v1set = set(match_keys("v1"))
rows = []
rec = defaultdict(lambda: np.zeros(2))
pos = []
how = Counter()
for k in match_keys("v2"):
    if k not in sc or TEAM not in sc[k]:
        continue
    r = load(k, "v2"); info = tba.match(k); tl = TL(r)
    al = "red" if TEAM in info.red else "blue"
    tm = info.teams(al)
    if not all(t in sc[k] for t in tm):
        continue
    f = np.array([r["teams"][str(t)]["fuel"] for t in tm]); s = np.array([sc[k][t] for t in tm])
    i = tm.index(TEAM)
    sig = team_signals(r, info)
    # credit sources
    src = np.zeros(3)
    for t, n in A.increments_from_counts(tl.counts.get(al) or []):
        mt = tl.match_time(t - 2.0)
        per = A.period_of(mt) if mt is not None else "x"
        p = []
        for team in tm:
            ts = sig[team]
            z = ts.zone_frac(t - A.LAG_HI, t - A.LAG_LO) if (A.AUTO_ZONE_GATE or per != "auto") else 1.0
            s_ = float((A.kernel(t - ts.shots) * ts.shot_w).sum()) if len(ts.shots) else 0.0
            vis = ts.vis_frac(t - A.LAG_HI, t - A.LAG_LO)
            p.append((z * s_, z * A.PRIOR * vis, z * A.HIDDEN * (1 - vis)))
        p = np.array(p)
        if p.sum() > 0:
            src += n * p[i] / p.sum()
    dets = json.loads((DIRS["v2"] / k / "dets.json").read_text())
    boxes = defaultdict(list)
    for d in dets:
        if d[7] is not None:
            boxes[d[7]].append((d[0], d[1]))
    for team in tm:
        rows_t = [x for x in bym[k].get(team, []) if time_valid(x)]
        L = np.array([q["t"] for q in r["shots"] if q["team"] == team and q["toward"]])
        bt = sorted(boxes.get(team, []))
        btt = np.array([b[0] for b in bt])
        pts = path_video_times(r["paths"].get(str(team), []), tl)
        for x in rows_t:
            for st in tl.auto_start + x.shots - 0.8:
                det = bool(len(L) and np.min(np.abs(L - st)) <= 1.0)
                seen = set()
                if len(btt):
                    j0, j1 = np.searchsorted(btt, st - 0.15), np.searchsorted(btt, st + 0.15)
                    seen = {bt[j][1] for j in range(j0, j1)}
                cov = "own_end" if f"{al}_end" in seen else ("main/opp only" if seen else "no box")
                who = "5026" if team == TEAM else "others"
                rec[(who, cov)] += (1 / len(rows_t), det / len(rows_t))
                rec[(who, "all")] += (1 / len(rows_t), det / len(rows_t))
                if team == TEAM and len(pts):
                    q = P._track_position(pts, st)
                    if q is not None:
                        pos.append((q[0] if al == "blue" else F.L - q[0], q[1] if al == "blue" else F.W - q[1], det))
    for q in r["shots"]:
        if q["team"] == TEAM and q["toward"]:
            how[(q["view"].replace(al, "own").replace("red", "opp").replace("blue", "opp"), q["how"])] += 1
    each = [x.n for x in bym[k].get(TEAM, [])]
    rows.append((k, al, round(float(f[i] / f.sum()), 2), round(float(s[i] / s.sum()), 2), int(f[i]), each,
                 np.round(src / max(src.sum(), 1e-9), 2).tolist(), "v2" if k in v1set else "v1-scan"))
print("5026 per scouted cc qual: video share, scout share, video fuel, scout counts, credit L/P/H, scan")
for x in rows:
    print("  ", x)
d = np.array([x[2] - x[3] for x in rows])
print(f"   mean share gap video - scout {d.mean():+.3f} (n={len(d)}); v2-scan matches {np.mean([x[2] - x[3] for x in rows if x[7] == 'v2']):+.3f}")
print("scouted-shot recall (credited launch within +-1 s):")
for (who, cov), (n, dd) in sorted(rec.items()):
    print(f"   {who:6s} {cov:14s} balls {n:6.0f} detected {100 * dd / n:5.1f} %")
pos = np.array(pos)
print(f"5026 shooting positions (own-alliance frame: x from own wall, y from own right?): x q10/50/90 "
      f"{np.round(np.quantile(pos[:, 0], [.1, .5, .9]), 1)}, y q10/50/90 {np.round(np.quantile(pos[:, 1], [.1, .5, .9]), 1)}")
for lo, hi in ((0, 2.7), (2.7, 5.4), (5.4, 8.1)):
    m = (pos[:, 1] >= lo) & (pos[:, 1] < hi)
    print(f"   y {lo}-{hi}: {m.sum():.0f} balls, detected {100 * pos[m, 2].mean() if m.any() else 0:.0f} %")
print("5026 credited launches by (view, link):", dict(how))
