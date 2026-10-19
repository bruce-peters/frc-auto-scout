"""Paired comparison of two nested-CV runs (fit_cv.py --out JSON, same spec name) on their datasets: per-match
bootstrap of the change in share error (TV) and team-match fuel MAE, B minus A.

  .venv/Scripts/python data/exp_attrmodel/tools/paired_cv.py A.json A.pkl B.json B.pkl SPEC
"""
import json
import pickle
import sys

import numpy as np


def load(js, pk, spec):
    m = json.load(open(js))["models"][spec]
    ds = pickle.load(open(pk, "rb"))
    sc = [r for r in ds["recs"] if r.get("overlay") and "scout_n" in r and r.get("tba_total")
          and r["video_share"] is not None]
    return np.array(m["pred"]), sc


ja, pa, jb, pb, spec = sys.argv[1:6]
A, sa = load(ja, pa, spec)
B, sb = load(jb, pb, spec)
ka = [(r["key"], r["al"]) for r in sa]
kb = {(r["key"], r["al"]): i for i, r in enumerate(sb)}
idx = [kb[k] for k in ka if k in kb]
keep = [i for i, k in enumerate(ka) if k in kb]
A, B, sa = A[keep], B[idx], [sa[i] for i in keep]
ref = np.array([r["scout_n"] / r["scout_n"].sum() for r in sa])
tot = np.array([r["tba_total"] for r in sa], float)
dtv = 0.5 * np.abs(B - ref).sum(1) - 0.5 * np.abs(A - ref).sum(1)
dm = (np.abs(B - ref) * tot[:, None]).mean(1) - (np.abs(A - ref) * tot[:, None]).mean(1)
mk = np.array([r["key"] for r in sa])
uk = sorted(set(mk))
P = np.array([(dtv[mk == k].mean(), dm[mk == k].mean(), (mk == k).sum()) for k in uk])
rng = np.random.default_rng(0)
bs = []
for _ in range(4000):
    j = rng.integers(0, len(uk), len(uk))
    w = P[j, 2]
    bs.append((np.average(P[j, 0], weights=w), np.average(P[j, 1], weights=w)))
bs = np.array(bs)
print(f"{len(sa)} alliances, {len(uk)} matches: dTV {np.average(P[:, 0], weights=P[:, 2]):+.4f} "
      f"{np.percentile(bs[:, 0], [2.5, 97.5]).round(4)}  dMAE {np.average(P[:, 1], weights=P[:, 2]):+.2f} "
      f"{np.percentile(bs[:, 1], [2.5, 97.5]).round(2)}  matches better {int((P[:, 1] < 0).sum())} / worse "
      f"{int((P[:, 1] > 0).sum())}")
