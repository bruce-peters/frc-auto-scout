import sys
import numpy as np
from common import load, match_keys, resplit, tba
src = sys.argv[1] if len(sys.argv) > 1 else "v2"
bad = []
for k in match_keys(src):
    r = load(k, src); info = tba.match(k)
    fu = resplit(r, info)
    for t in info.red + info.blue:
        d = fu[t] - r["teams"][str(t)]["fuel"]
        bad.append((abs(d), k, t, round(fu[t], 1), r["teams"][str(t)]["fuel"]))
bad.sort(reverse=True)
a = np.array([b[0] for b in bad])
print(f"n={len(a)} max {a.max():.2f} mean {a.mean():.3f} >1 fuel: {(a > 1).sum()}")
for b in bad[:8]:
    print(b)
