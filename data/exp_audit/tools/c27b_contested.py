"""Cycle 27b: volleys measured against the scoreboard, and what happens when a volley is contested.

1) Sole-evidence bursts (no other alliance robot has a credited launch in [first - 1, last + 1]): credited
   launches per COUNTER ball (counter balls in [first + 0.8, last + 5.5]) by scouted burst size. If flat, launch
   detection is proportional to scored balls and the scouted-burst 'undercount' is a tap-timing artifact.
2) Contested bursts: the robot's share of the alliance's credited launches in the window vs its share of the
   alliance's scouted shots in the same window (all three robots, time_valid rows), by burst size. Mean
   (credited share - scouted share) < 0 for big bursts = volley shooters under-credited when contested.
3) How often bursts are contested, by size.
"""
from collections import defaultdict

import numpy as np

from common import A, P, TL, load, match_keys, tba, time_valid
from autoscout import scouting

bym = scouting.by_match(scouting.load())
keys = [k for k in match_keys("v2") if k.startswith("2026cc_qm") and k in bym]
sole = defaultdict(lambda: np.zeros(3))      # class -> (w, credited, counter balls)
cont = defaultdict(list)                     # class -> (credited share - scouted share)
frac = defaultdict(lambda: np.zeros(2))      # class -> (bursts, contested)
CLS = lambda n: "1-4" if n < 5 else ("5-9" if n < 10 else ("10-19" if n < 20 else "20+"))
for k in keys:
    r = load(k, "v2"); info = tba.match(k); tl = TL(r)
    cred = defaultdict(list)
    for s in r["shots"]:
        if s["team"] is not None and s["toward"]:
            cred[s["team"]].append(s["t"])
    cred = {t: np.sort(v) for t, v in cred.items()}
    for al in ("red", "blue"):
        tm = info.teams(al)
        rows = {t: [x for x in bym[k].get(t, []) if time_valid(x)] for t in tm}
        inc = A.increments_from_counts(P._match_counts(tl, al, info.hub(al) or {})) if tl.counts.get(al) else []
        it = np.array([x[0] for x in inc]); iv = np.array([x[1] for x in inc])
        st_all = {t: [tl.auto_start + x.shots - 0.8 for x in rows[t]] for t in tm}
        full = all(rows.values())
        for team in tm:
            for ri, st in enumerate(st_all[team]):
                st = np.sort(st)
                if not len(st):
                    continue
                w = 1.0 / len(st_all[team])
                for b in np.split(st, np.where(np.diff(st) > 1.5)[0] + 1):
                    a0, a1 = b[0] - 1.0, b[-1] + 1.0
                    c = {u: int(((cred.get(u, np.zeros(0)) >= a0) & (cred.get(u, np.zeros(0)) <= a1)).sum()) for u in tm}
                    cl = CLS(len(b))
                    contested = any(c[u] > 0 for u in tm if u != team)
                    frac[cl] += (w, w * contested)
                    if not contested:
                        cn = float(iv[(it >= b[0] + 0.8) & (it <= b[-1] + 5.5)].sum()) if len(it) else 0.0
                        if cn > 0:
                            sole[cl] += (w, w * c[team], w * cn)
                    elif full and sum(c.values()) > 0:
                        # scouted shots of every alliance robot in the same window (mean over each robot's scouts)
                        sc = {u: np.mean([((s_ >= a0) & (s_ <= a1)).sum() for s_ in st_all[u]]) for u in tm}
                        if sum(sc.values()) > 0:
                            cont[cl].append((w, c[team] / sum(c.values()) - sc[team] / sum(sc.values()),
                                             c[team] / sum(c.values()), sc[team] / sum(sc.values())))
print("1) sole-evidence bursts: credited launches per counter ball")
for cl in ("1-4", "5-9", "10-19", "20+"):
    w, cr, cn = sole[cl]
    print(f"   {cl:6s} credited {cr / w:5.2f}/burst  counter balls {cn / w:5.1f}/burst  credited per counter ball {cr / max(cn, 1e-9):.3f}")
print("2) contested bursts: robot's credited share - its scouted share in the window")
for cl in ("1-4", "5-9", "10-19", "20+"):
    x = np.array(cont[cl])
    if len(x):
        print(f"   {cl:6s} n {x[:, 0].sum():6.0f}  mean diff {np.average(x[:, 1], weights=x[:, 0]):+.3f}  "
              f"(credited share {np.average(x[:, 2], weights=x[:, 0]):.2f}, scouted share {np.average(x[:, 3], weights=x[:, 0]):.2f})")
print("3) share of bursts contested (another alliance robot has a credited launch in the window)")
for cl in ("1-4", "5-9", "10-19", "20+"):
    w, c = frac[cl]
    print(f"   {cl:6s} bursts {w:6.0f}  contested {100 * c / w:5.1f} %")
