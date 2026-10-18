"""Cycle 16c: a team with NO bumper evidence all match (absent / dead / parked robot?) whose label the
three-slot identity model hands to fragments of its teammates' tracks.

Unread team: total own-number evidence < 0.5 over its labelled segments, while each teammate has >= 3.
Per unread team: its scouted share, video share, and how much teammate-read evidence sits on the segments
labelled with it. Counterfactuals (result.json re-split):
  relink : launches on its segments whose reads name a teammate (>= 1.0) go to that teammate
  absent : relink + the unread team gets no credit at all (weight 0)
Scored vs scouting and OPR.
"""
import copy
import json

import numpy as np

from common import A, AUDIT, TL, load, match_keys, resplit, scout_counts, scout_metrics, share_metrics, tba, team_signals

sc = scout_counts()
R, INFO, UN = {}, {}, {}
stats = []
for k in match_keys("v2"):
    r = load(k, "v2"); info = tba.match(k)
    R[k], INFO[k] = r, info
    for al in ("red", "blue"):
        tm = info.teams(al)
        ide = r["identity"][al]
        own = {t: sum(e["ocr"][i] for e in ide.values() if e["team"] == t) for i, t in enumerate(tm)}
        for i, t in enumerate(tm):
            mates = [u for u in tm if u != t]
            if own[t] < 0.5 and all(own[u] >= 3 for u in mates):
                foreign = {u: sum(e["ocr"][tm.index(u)] for e in ide.values() if e["team"] == t) for u in mates}
                UN.setdefault(k, []).append((al, t, foreign))
                f = np.array([r["teams"][str(x)]["fuel"] for x in tm])
                s = np.array([sc[k][x] for x in tm]) if k in sc and all(x in sc[k] for x in tm) else None
                stats.append({"key": k, "team": t, "video_share": round(float(f[i] / f.sum()), 3) if f.sum() else None,
                              "scout_share": round(float(s[i] / s.sum()), 3) if s is not None and s.sum() else None,
                              "foreign_reads": {u: round(v, 1) for u, v in foreign.items()}})
n_tm = sum(1 for k in R for _ in INFO[k].red + INFO[k].blue)
print(f"{len(R)} matches: unread team-matches {len(stats)} of {n_tm} ({100 * len(stats) / n_tm:.1f} %)")
ss = [x for x in stats if x["scout_share"] is not None]
print(f"   scouted ones: {len(ss)}; scout share mean {np.mean([x['scout_share'] for x in ss]):.3f}, video share mean "
      f"{np.mean([x['video_share'] for x in ss]):.3f}")
for x in ss:
    print("   ", x)
print("   unscouted examples:", [x for x in stats if x["scout_share"] is None][:6])

# counterfactuals
ORIG_SPLIT = A.split_increments
base, rel, absn = {}, {}, {}
for k in R:
    r, info = R[k], INFO[k]
    base[k] = resplit(r, info)
    if k not in UN:
        rel[k] = absn[k] = base[k]
        continue
    r2 = copy.deepcopy(r)
    for al, t, foreign in UN[k]:
        tm = info.teams(al)
        segs = [e for e in r["identity"][al].values() if e["team"] == t]
        for e in segs:
            o = np.array(e["ocr"])
            if o.sum() < 1.0:
                continue
            to = tm[int(np.argmax(o))]
            if to == t:
                continue
            for s, s2 in zip(r["shots"], r2["shots"]):
                if s["team"] == t and e["t0"] - 0.3 <= s["t"] <= e["t1"] + 0.3:
                    s2["team"] = to
    rel[k] = resplit(r2, info)
    absent = {t for _, t, _ in UN[k]}

    def split_wo(inc, teams, tmt, log=None, absent=absent):
        keep = [ts for ts in teams if ts.team not in absent]
        out = ORIG_SPLIT(inc, keep, tmt, log=log)
        for ts in teams:
            out.setdefault(ts.team, {})
        return out
    A.split_increments = split_wo
    absn[k] = resplit(r2, info)
    A.split_increments = ORIG_SPLIT
for name, pm in (("base", base), ("relink", rel), ("absent", absn)):
    print(f"{name:7s}", share_metrics(pm), scout_metrics(pm))
json.dump({"unread": stats}, open(AUDIT / "c16c_unread.json", "w"), indent=1, default=float)
