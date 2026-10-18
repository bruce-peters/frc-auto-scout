"""Cycle 27c: does launch detection saturate at high scoring rates? Scoreboard-only ruler, all matches.

Counter bursts per alliance (increments <= 1.5 s apart). For each burst, credited launches of each alliance
robot in [start - 5.5, end - 0.8]. Sole bursts: exactly one robot has launches -> that robot scored the burst.
Rate = balls / (burst duration + 0.33 s). Credited launches per counter ball by rate bin; also how often a
burst is contested (>= 2 robots with launches), by rate.
"""
from collections import defaultdict

import numpy as np

from common import A, P, TL, load, match_keys, tba

sole = defaultdict(lambda: np.zeros(3))  # rate bin -> (bursts, balls, launches)
cont = defaultdict(lambda: np.zeros(2))  # rate bin -> (bursts, contested)
BINS = [0, 2, 4, 6, 9, 13, 1e9]
lab = lambda r: next(f"{a}-{b}" for a, b in zip(BINS[:-1], BINS[1:]) if a <= r < b)
for k in match_keys("v2"):
    r = load(k, "v2"); info = tba.match(k); tl = TL(r)
    cred = defaultdict(list)
    for s in r["shots"]:
        if s["team"] is not None and s["toward"]:
            cred[s["team"]].append(s["t"])
    cred = {t: np.sort(v) for t, v in cred.items()}
    for al in ("red", "blue"):
        if not tl.counts.get(al):
            continue
        inc = A.increments_from_counts(P._match_counts(tl, al, info.hub(al) or {}))
        if not inc:
            continue
        t = np.array([x[0] for x in inc]); n = np.array([x[1] for x in inc])
        for b in np.split(np.arange(len(t)), np.where(np.diff(t) > 1.5)[0] + 1):
            bt, bn = t[b], n[b]
            balls = bn.sum()
            if balls < 3:
                continue
            rate = balls / (bt[-1] - bt[0] + 0.33)
            L = {u: int(((cred.get(u, np.zeros(0)) >= bt[0] - 5.5) & (cred.get(u, np.zeros(0)) <= bt[-1] - 0.8)).sum())
                 for u in info.teams(al)}
            act = [u for u, c in L.items() if c > 0]
            lb = lab(rate)
            cont[lb] += (1, len(act) >= 2)
            if len(act) == 1:
                sole[lb] += (1, balls, L[act[0]])
print("sole bursts (one robot with launches): credited launches per counter ball by scoring rate")
for a, b in zip(BINS[:-1], BINS[1:]):
    lb = f"{a}-{b}"
    nb, balls, la = sole[lb]
    c = cont[lb]
    print(f"   {lb:6s} balls/s: sole bursts {nb:5.0f}, balls {balls:6.0f}, launches per ball {la / max(balls, 1):.3f} | "
          f"contested {100 * c[1] / max(c[0], 1):4.1f} % of {c[0]:.0f} bursts")
