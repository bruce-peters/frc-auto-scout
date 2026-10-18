"""Cycle 16: the new-cc gap (scout share agreement new vs original cc matches) - concentrated, 5026-related, or noise?

Groups of scouted cc alliances (all three robots scouted, current output/):
  orig_5026   : original 14 cc matches, alliance with 5026
  orig_other  : original 14 cc matches, the other alliance
  new         : cc matches processed later today
Per group: mean TV share error vs scouting, scout-vs-scout agreement (TV between two scouts' shares),
TV scout-vs-OPR (how 'normal' the alliance is), concentration (share of squared error in the worst 3
alliances), bootstrap CI of the new - orig difference.
"""
import json

import numpy as np

from common import AUDIT, load, match_keys, opr, tba
from autoscout import scouting

ORIG = set(match_keys("v1"))
bym = scouting.by_match(scouting.load())
rows = []
for k in match_keys("v2"):
    if not k.startswith("2026cc_qm") or k not in bym:
        continue
    r = load(k, "v2"); info = tba.match(k); op = opr(info.event_key)
    for al in ("red", "blue"):
        tm = info.teams(al)
        rs = [bym[k].get(t, []) for t in tm]
        if not all(rs):
            continue
        s = np.array([np.mean([x.n for x in rr]) for rr in rs]); f = np.array([r["teams"][str(t)]["fuel"] for t in tm])
        o = np.array([op.get(t, np.nan) for t in tm])
        if s.sum() <= 0 or f.sum() <= 0:
            continue
        s_, f_, o_ = s / s.sum(), f / f.sum(), o / o.sum()
        tv = lambda a, b: float(np.abs(a - b).sum() / 2)
        ab = None
        if all(len(rr) >= 2 for rr in rs):
            a = np.array([rr[0].n for rr in rs], float); b = np.array([rr[1].n for rr in rs], float)
            if a.sum() and b.sum():
                ab = tv(a / a.sum(), b / b.sum())
        grp = ("orig_5026" if 5026 in tm else "orig_other") if k in ORIG else "new"
        rows.append({"key": k, "al": al, "teams": tm, "grp": grp, "err": tv(f_, s_), "scout_ab": ab, "scout_opr": tv(s_, o_),
                     "video": np.round(f_, 2).tolist(), "scout": np.round(s_, 2).tolist()})
for g in ("orig_5026", "orig_other", "new"):
    x = [r for r in rows if r["grp"] == g]
    if not x:
        continue
    e = np.array([r["err"] for r in x]); ab = [r["scout_ab"] for r in x if r["scout_ab"] is not None]
    srt = np.sort(e ** 2)[::-1]
    print(f"{g:11s} alliances {len(x):3d}  err vs scout mean {e.mean():.3f} median {np.median(e):.3f}  worst-3 hold "
          f"{100 * srt[:3].sum() / srt.sum():.0f} % of sq err;  scout A-vs-B {np.mean(ab) if ab else float('nan'):.3f} (n={len(ab)});  "
          f"scout-vs-OPR {np.mean([r['scout_opr'] for r in x]):.3f}")
orig = np.array([r["err"] for r in rows if r["grp"] != "new"]); new = np.array([r["err"] for r in rows if r["grp"] == "new"])
rng = np.random.default_rng(0)
d = [rng.choice(new, len(new)).mean() - rng.choice(orig, len(orig)).mean() for _ in range(5000)]
print(f"new - orig mean err {new.mean() - orig.mean():+.3f}, 95 % CI [{np.quantile(d, .025):+.3f}, {np.quantile(d, .975):+.3f}]")
print("worst alliances:")
for r in sorted(rows, key=lambda r: -r["err"])[:10]:
    print(f"   {r['grp']:10s} {r['key']:13s} {r['al']:4s} {r['teams']} err {r['err']:.2f} video {r['video']} scout {r['scout']} scoutAB {r['scout_ab']}")
json.dump(rows, open(AUDIT / "c16_newcc_gap.json", "w"), indent=1, default=float)
