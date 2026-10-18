"""Cycle 16b: robots the scouts saw shoot (almost) nothing - how much credit do they get, from what?

Non-shooter: scouted share < 0.06 with every scout <= 15 shots. For each: video share, credit sources
(launch / prior / hidden), its credited launches (count, link type, view, zone at launch, toward), and the
share of its launch-evidence credit that coincides with a teammate's launch within +-1 s. Frequency of
non-shooters per group (original vs new cc) and the error they cause.
"""
import json
from collections import Counter

import numpy as np

from common import A, AUDIT, TL, load, match_keys, tba, team_signals
from autoscout import scouting

ORIG = set(match_keys("v1"))
bym = scouting.by_match(scouting.load())
rows = []
grp_n = Counter()
for k in match_keys("v2"):
    if not k.startswith("2026cc_qm") or k not in bym:
        continue
    r = load(k, "v2"); info = tba.match(k); tl = TL(r)
    sig = None
    for al in ("red", "blue"):
        tm = info.teams(al)
        rs = [bym[k].get(t, []) for t in tm]
        if not all(rs):
            continue
        g = "orig" if k in ORIG else "new"
        grp_n[g] += 1
        s = np.array([np.mean([x.n for x in rr]) for rr in rs]); f = np.array([r["teams"][str(t)]["fuel"] for t in tm])
        if s.sum() <= 0 or f.sum() <= 0:
            continue
        sig = sig or team_signals(r, info)
        src = np.zeros((3, 3))
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
                src += n * p / p.sum()
        for i, t in enumerate(tm):
            if s[i] / s.sum() < 0.06 and all(x.n <= 15 for x in rs[i]):
                sh = [q for q in r["shots"] if q["team"] == t and q["toward"]]
                mates = [q["t"] for q in r["shots"] if q["team"] in tm and q["team"] != t and q["toward"]]
                coinc = sum(1 for q in sh if mates and min(abs(q["t"] - m) for m in mates) <= 1.0)
                zl = []
                for q in sh:
                    ts = sig[t]
                    m = (ts.zone_t >= q["t"] - 0.5) & (ts.zone_t <= q["t"] + 0.5)
                    v = ts.zone_v[m]; v = v[~np.isnan(v)]
                    zl.append(float(v.mean()) if len(v) else np.nan)
                rows.append({"grp": g, "key": k, "al": al, "team": t, "scout": [x.n for x in rs[i]],
                             "video_share": round(float(f[i] / f.sum()), 3), "fuel": round(float(f[i]), 1),
                             "src_LPH": np.round(src[i] / max(src[i].sum(), 1e-9), 2).tolist(),
                             "launches": len(sh), "how": dict(Counter(q["how"] for q in sh)),
                             "views": dict(Counter(q["view"] for q in sh)),
                             "inzone_launch_frac": round(float(np.nanmean(np.array(zl) >= 0.5)) if zl else 0.0, 2),
                             "coincide_with_mate": coinc,
                             "vis": round(float(sig[t].vis_v.mean()) if len(sig[t].vis_v) else 0.0, 2)})
for g in ("orig", "new"):
    x = [r for r in rows if r["grp"] == g]
    print(f"{g}: {grp_n[g]} scouted alliances, non-shooters {len(x)} ({len(x) / max(grp_n[g], 1):.2f} per alliance); "
          f"their video share mean {np.mean([r['video_share'] for r in x]) if x else 0:.3f}")
    if x:
        L = np.array([r["src_LPH"] for r in x])
        print(f"    credit source mean L/P/H {np.round(L.mean(0), 2).tolist()}; launches median {np.median([r['launches'] for r in x]):.0f}; "
              f"in-zone launch frac median {np.median([r['inzone_launch_frac'] for r in x]):.2f}; "
              f"launches coinciding with a teammate's (+-1 s) {sum(r['coincide_with_mate'] for r in x)}/{sum(r['launches'] for r in x)}")
for r in sorted(rows, key=lambda r: -r["video_share"]):
    print("  ", r)
json.dump(rows, open(AUDIT / "c16b_nonshooters.json", "w"), indent=1, default=float)
