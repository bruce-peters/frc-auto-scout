"""Cycle 3c: per-robot launch recall (detected launch weight per scouted ball, while visible in zone) and
what it depends on: scouted shooting rate, where the robot shoots from, which camera sees its launches,
team (repeatable across matches?). Also: how the per-robot gain model attribution.split_nnls does.
Usage: c3c_recall.py [SRC]  (default cc_v2)
"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np

from common import A, AUDIT, DIRS, F, P, TL, time_valid, load, path_video_times, resplit, scout_metrics, share_metrics, tba, team_signals
from autoscout import scouting

src = sys.argv[1] if len(sys.argv) > 1 else "cc_v2"
TAP = 0.8
bym = scouting.by_match(scouting.load())
rows = []
R, INFO, SIG = {}, {}, {}
import os
EXCL = set(os.environ.get("EXCLUDE", "").split(","))
for key, teams in sorted(bym.items()):
    if key in EXCL or not (DIRS[src] / key / "result.json").exists():
        continue
    r = load(key, src); info = tba.match(key); tl = TL(r)
    R[key], INFO[key] = r, info
    sig = SIG[key] = team_signals(r, info)
    for al in ("red", "blue"):
        for t in info.teams(al):
            srs = [x for x in teams.get(t, []) if time_valid(x)]
            ts = sig[t]
            if not srs or not len(ts.zone_t):
                continue
            sc_all = np.sort(np.concatenate([tl.auto_start + x.shots - TAP for x in srs]))
            w_sc = 1.0 / len(srs)
            sh = [s for s in r["shots"] if s["team"] == t and s["toward"]]
            lt = np.array([s["t"] for s in sh]); lc = np.array([s["conf"] for s in sh])
            pts = path_video_times(r["paths"].get(str(t), []), tl)
            balls = lw = 0.0
            xy = []
            for g, z, v in zip(ts.zone_t, ts.zone_v, ts.vis_v):
                if z != 1.0 or v < 0.5:
                    continue
                nb = w_sc * ((sc_all >= g - 0.25) & (sc_all < g + 0.25)).sum()
                balls += nb
                if len(lt):
                    lw += lc[(lt >= g - 0.25) & (lt < g + 0.25)].sum()
                if nb > 0 and len(pts):
                    p = P._track_position(pts, g)
                    if p is not None:
                        xy.append((p[0], p[1], nb))
            if balls < 15:
                continue
            # scouted shooting rate: balls per second over the 1 s bins that have any scouted ball
            bins = np.floor(sc_all).astype(int)
            cnt = Counter(bins)
            act = np.array(list(cnt.values())) * w_sc
            xy = np.array(xy) if xy else np.zeros((0, 3))
            own_wall_d = (xy[:, 0] if al == "blue" else F.L - xy[:, 0]) if len(xy) else np.zeros(0)
            views = Counter(s["view"] for s in sh)
            rows.append({"match": key, "team": t, "alliance": al, "recall": lw / balls, "balls_vis": balls,
                         "scout_total": float(np.mean([x.n for x in srs])),
                         "rate_active": float(np.average(act, weights=act)),  # ball-weighted balls/s in active seconds
                         "wall_d": float(np.average(own_wall_d, weights=xy[:, 2])) if len(xy) else np.nan,
                         "y": float(np.average(xy[:, 1], weights=xy[:, 2])) if len(xy) else np.nan,
                         "frac_main": views.get("main", 0) / max(1, len(sh)),
                         "frac_own_end": views.get(f"{al}_end", 0) / max(1, len(sh)),
                         "frac_opp_end": views.get(("red" if al == "blue" else "blue") + "_end", 0) / max(1, len(sh)),
                         "frac_proj": sum(1 for s in sh if s["how"] != "box") / max(1, len(sh)),
                         "vis": float(ts.vis_v.mean())})
rec = np.array([x["recall"] for x in rows])
print(f"{len(rows)} scouted robot-matches ({src}); recall median {np.median(rec):.3f}")
for col in ("rate_active", "scout_total", "wall_d", "y", "frac_main", "frac_own_end", "frac_opp_end", "frac_proj", "vis"):
    v = np.array([x[col] for x in rows], float)
    ok = np.isfinite(v)
    rho = np.corrcoef(np.argsort(np.argsort(v[ok])), np.argsort(np.argsort(rec[ok])))[0, 1]
    print(f"   spearman(recall, {col:12s}) = {rho:+.2f}   ({col} median {np.nanmedian(v):.2f})")
by_team = defaultdict(list)
for x in rows:
    by_team[x["team"]].append(x["recall"])
multi = {t: v for t, v in by_team.items() if len(v) >= 2}
print(f"\nteams with >= 2 scouted matches: {len(multi)}")
# repeatability: share of recall variance explained by team
allv = np.concatenate(list(multi.values()))
within = np.mean([np.var(v) for v in multi.values()])
print(f"   recall variance: total {np.var(allv):.4f}, mean within-team {within:.4f}  -> team explains "
      f"{100 * (1 - within / np.var(allv)):.0f} %")
for t, v in sorted(multi.items(), key=lambda kv: np.mean(kv[1])):
    print(f"   {t:5d} recall {' '.join(f'{x:.2f}' for x in v)}")
# terciles by shooting rate
order = np.argsort([x["rate_active"] for x in rows])
for name, idx in (("slowest third", order[:len(order) // 3]), ("middle third", order[len(order) // 3:2 * len(order) // 3]),
                  ("fastest third", order[2 * len(order) // 3:])):
    print(f"   {name:14s} rate {np.median([rows[i]['rate_active'] for i in idx]):.1f} balls/s  recall median "
          f"{np.median(rec[idx]):.3f}")

# split_nnls vs split_increments
pm = {k: resplit(R[k], INFO[k], signals=SIG[k]) for k in R}
orig = A.split_increments
A.split_increments = lambda inc, teams, tmt, log=None: A.split_nnls(inc, teams, tmt, log=log)
pn = {k: resplit(R[k], INFO[k], signals=SIG[k]) for k in R}
A.split_increments = orig
print("\nsplit_increments:", scout_metrics(pm), share_metrics(pm))
print("split_nnls      :", scout_metrics(pn), share_metrics(pn))
# oracle: what if each robot's launch weights were divided by its true (scouted) recall?
json.dump(rows, open(AUDIT / f"c3c_recall_{src}{'_excl' if EXCL - {''} else ''}.json", "w"), indent=1, default=float)
