"""Paired bootstrap over matches: share-corr / top-scorer difference between two fuel tables."""
import json, sys
import numpy as np
from common import AUDIT, opr, tba
d = json.load(open(AUDIT / sys.argv[1]))
a_name, b_name = sys.argv[2], sys.argv[3]
F = {n: {k: {int(t): v for t, v in f.items()} for k, f in d["fuel"][n].items()} for n in (a_name, b_name)}
keys = sorted(F[a_name])
rows = {}
for k in keys:
    info = tba.match(k); op = opr(info.event_key)
    rr = []
    for al in ("red", "blue"):
        tm = info.teams(al)
        o = np.array([op.get(t, np.nan) for t in tm])
        fa = np.array([F[a_name][k][t] for t in tm]); fb = np.array([F[b_name][k][t] for t in tm])
        if fa.sum() < 40 or not np.all(np.isfinite(o)) or o.sum() <= 0:
            continue
        rr.append((fa / fa.sum(), fb / fb.sum(), o / o.sum()))
    rows[k] = rr
def metr(ks):
    A_, B_, O_ = [], [], []
    ta = tb = 0
    for k in ks:
        for a, b, o in rows[k]:
            A_ += a.tolist(); B_ += b.tolist(); O_ += o.tolist()
            ta += int(np.argmax(a) == np.argmax(o)); tb += int(np.argmax(b) == np.argmax(o))
    return np.corrcoef(A_, O_)[0, 1], np.corrcoef(B_, O_)[0, 1], ta, tb
ca, cb, ta, tb = metr(keys)
rng = np.random.default_rng(0)
dc, dt = [], []
for _ in range(2000):
    ks = rng.choice(keys, len(keys))
    x = metr(ks); dc.append(x[1] - x[0]); dt.append(x[3] - x[2])
dc, dt = np.array(dc), np.array(dt)
print(f"{len(keys)} matches: corr {a_name} {ca:.3f} -> {b_name} {cb:.3f} (diff {cb - ca:+.3f}, 95% CI [{np.quantile(dc, .025):+.3f}, "
      f"{np.quantile(dc, .975):+.3f}], P(better) {np.mean(dc > 0):.3f}); top {ta} -> {tb} (CI [{np.quantile(dt, .025):+.0f}, {np.quantile(dt, .975):+.0f}])")
