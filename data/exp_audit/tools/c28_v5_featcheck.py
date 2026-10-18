"""Cycle 28: my v5 dataset (build_ds_vol3) must equal the main session's ds_v5.pkl in every feature but s_beq."""
import pickle
import sys

import numpy as np

a, b = (pickle.load(open(p, "rb")) for p in sys.argv[1:3])
fa, fb = a["feats"], b["feats"]
assert fa[:-1] == fb and fa[-1] == "s_beq", (fa, fb)
ra, rb = a["recs"], b["recs"]
assert [(r["key"], r.get("alliance", r.get("al"))) for r in ra] == [(r["key"], r.get("alliance", r.get("al"))) for r in rb]
mx, n, keys_diff = 0.0, 0, set()
for x, y in zip(ra, rb):
    for k in y:
        if k == "X":
            d = np.abs(np.asarray(x["X"])[..., :-1] - np.asarray(y["X"])) if len(y["X"]) else np.zeros(0)
            mx = max(mx, float(d.max()) if d.size else 0.0)
        else:
            try:
                ya, xa = np.asarray(y[k], dtype=float), np.asarray(x.get(k), dtype=float)
                same = xa.shape == ya.shape and np.allclose(xa, ya, equal_nan=True, rtol=0, atol=1e-9)
            except (TypeError, ValueError):
                same = repr(x.get(k)) == repr(y[k])
            if not same:
                keys_diff.add(k)
    n += 1
print(f"{n} alliance records, max |X diff| (all feats but s_beq) {mx:.2e}, other fields differing: {sorted(keys_diff) or 'none'}")
