"""Cycle 3: credit from HIDDEN pseudo-shots (attribution.HIDDEN) and the PRIOR.

1. split every counter increment's weight into launch evidence / prior / hidden pseudo-shots (per event)
2. how 'hidden' a robot is (fraction of 0.5 s grid samples without a fused fix within 0.3 s), per event
   and per team-match; which team-matches get most of their fuel from hidden credit
3. HIDDEN / PRIOR sweep -> share metrics vs fuel OPR (+ vs scouting where Chezy quals are present)
Usage: c3_hidden.py [v2|v1]
"""
import json
import sys
from collections import defaultdict

import numpy as np

from common import A, AUDIT, TL, load, match_keys, resplit, scout_metrics, share_metrics, tba, team_signals

src = sys.argv[1] if len(sys.argv) > 1 else "v2"
keys = match_keys(src)
R = {k: load(k, src) for k in keys}
INFO = {k: tba.match(k) for k in keys}
SIG = {k: team_signals(R[k], INFO[k]) for k in keys}
ev_of = lambda k: INFO[k].event_key

# ---------- 1/2. decomposition + visibility
dec = defaultdict(lambda: np.zeros(4))  # event -> balls via launch, prior, hidden, even-split
team_rows = []
for k in keys:
    r, info = R[k], INFO[k]
    tl = TL(r)
    for al in ("red", "blue"):
        teams = info.teams(al)
        sig = [SIG[k][t] for t in teams]
        per_team = np.zeros((3, 3))  # team x (launch, prior, hidden) balls
        if tl.counts.get(al):
            for t, n in A.increments_from_counts(tl.counts[al]):
                parts = []
                for ts in sig:
                    z = ts.zone_frac(t - A.LAG_HI, t - A.LAG_LO)
                    s = float((A.kernel(t - ts.shots) * ts.shot_w).sum()) if len(ts.shots) else 0.0
                    vis = ts.vis_frac(t - A.LAG_HI, t - A.LAG_LO)
                    parts.append((z * s, z * A.PRIOR * vis, z * A.HIDDEN * (1 - vis)))
                p = np.array(parts)
                if p.sum() <= 1e-9:
                    dec[ev_of(k)][3] += n
                    continue
                dec[ev_of(k)][:3] += n * p.sum(0) / p.sum()
                per_team += n * p / p.sum()
        for i, (t, ts) in enumerate(zip(teams, sig)):
            vis = float(ts.vis_v.mean()) if len(ts.vis_v) else 0.0
            tot = per_team[i].sum()
            team_rows.append({"match": k, "team": t, "alliance": al, "vis": vis,
                              "fuel_raw": float(tot), "from_launch": float(per_team[i, 0]),
                              "from_prior": float(per_team[i, 1]), "from_hidden": float(per_team[i, 2]),
                              "launches": int(len(ts.shots))})
print(f"{len(keys)} matches ({src})\ncredit sources by event (% of balls): launch / prior / hidden / even-split")
for e, v in sorted(dec.items()):
    print(f"   {e:12s} {100 * v[0] / v.sum():5.1f} {100 * v[1] / v.sum():5.1f} {100 * v[2] / v.sum():5.1f} "
          f"{100 * v[3] / v.sum():5.1f}   ({v.sum():.0f} balls)")
allv = sum(dec.values())
print(f"   {'ALL':12s} {100 * allv[0] / allv.sum():5.1f} {100 * allv[1] / allv.sum():5.1f} "
      f"{100 * allv[2] / allv.sum():5.1f} {100 * allv[3] / allv.sum():5.1f}")
print("\nvisible fraction of match per team-match (median / q10 / q90), by event:")
byev = defaultdict(list)
for x in team_rows:
    byev[ev_of(x["match"])].append(x["vis"])
for e, v in sorted(byev.items()):
    print(f"   {e:12s} {np.median(v):.2f} {np.quantile(v, .1):.2f} {np.quantile(v, .9):.2f}  (n={len(v)})")
hid = sorted(team_rows, key=lambda x: -x["from_hidden"])
print("\nteam-matches with most hidden-sourced fuel (unscaled balls):")
for x in hid[:15]:
    print(f"   {x['match']:16s} {x['team']:5d} {x['alliance']:4s} vis {x['vis']:.2f} fuel {x['fuel_raw']:5.0f} "
          f"hidden {x['from_hidden']:5.0f} launch {x['from_launch']:5.0f} prior {x['from_prior']:4.0f} "
          f"launches {x['launches']}")
fr = np.array([x["from_hidden"] / max(x["fuel_raw"], 1e-9) for x in team_rows if x["fuel_raw"] > 5])
print(f"   team-matches where > 50 % of the fuel is hidden credit: {np.mean(fr > 0.5) * 100:.1f} %;  > 25 %: "
      f"{np.mean(fr > 0.25) * 100:.1f} %")

# ---------- 3. sweep
sweep = []
H0, P0 = A.HIDDEN, A.PRIOR
print("\nHIDDEN / PRIOR sweep:")
for h in (0.0, 0.5, 1.0, 2.0, 3.0):
    for p in (0.05, 0.12, 0.3):
        A.HIDDEN, A.PRIOR = h, p
        pm = {k: resplit(R[k], INFO[k], signals=SIG[k]) for k in keys}
        m = share_metrics(pm)
        s = scout_metrics(pm)
        m.update({"HIDDEN": h, "PRIOR": p, **(s or {})})
        sweep.append(m)
        print(f"   HIDDEN {h:3.1f} PRIOR {p:4.2f}: corr {m['corr']:.3f} top {m['top']}/{m['alliances']} MAE {m['mae']:.1f}"
              + (f"  | scout corr {s['scout_corr']:.3f} mae {s['scout_mae_share']:.3f} (n={s['n']})" if s else ""))
A.HIDDEN, A.PRIOR = H0, P0
json.dump({"decomp": {e: v.tolist() for e, v in dec.items()}, "team_rows": team_rows, "sweep": sweep},
          open(AUDIT / f"c3_hidden_{src}.json", "w"), indent=1)
