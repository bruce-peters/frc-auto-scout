"""Paired comparison of nested-CV credit-model refits (fit_cv.py --out JSON + build_ds.py dataset) by alliance."""
import json, pickle, sys
from collections import defaultdict
import numpy as np
spec = "lin/all+dist+mlprior"


def load(tag, root="data/exp_track"):
    cv = json.load(open(f"{root}/cv_{tag}.json")); ds = pickle.load(open(f"{root}/ds_{tag}.pkl", "rb"))
    recs = [r for r in ds["recs"] if r.get("overlay") and "scout_n" in r]
    pred = np.array(cv["models"][spec]["pred"], float)
    assert len(recs) == len(pred), (tag, len(recs), len(pred))
    out = {}
    for r, p in zip(recs, pred):
        ref = r["scout_n"] / r["scout_n"].sum()
        tot = r.get("tba_total") or float(np.sum(r["n"]))
        out[(r["key"], r["al"])] = dict(tv=0.5 * np.abs(p - ref).sum(), mae=float(np.mean(np.abs(p - ref) * tot)),
                                        top=float(np.argmax(p) == np.argmax(ref)))
    return out


base_tag, tags = sys.argv[1], sys.argv[2:]
B = load(base_tag)
rng = np.random.default_rng(0)
for tag in tags:
    V = load(tag)
    ks = sorted(set(B) & set(V))
    for m in ("tv", "mae", "top"):
        d = np.array([V[k][m] - B[k][m] for k in ks])
        by = defaultdict(list)
        for k, x in zip(ks, d):
            by[k[0]].append(x)
        mk = sorted(by)
        bs = [np.mean([x for kk in rng.choice(mk, len(mk)) for x in by[kk]]) for _ in range(4000)]
        print(f"{tag:12s} vs {base_tag}: d{m:4s} {d.mean():+.4f}  95% CI [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]"
              f"  ({len(ks)} alliances, {len(mk)} matches; {base_tag} mean {np.mean([B[k][m] for k in ks]):.3f})")
