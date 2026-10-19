"""Diagnostic: launch point minus the scouted shooter's fused position, on the hidden truth set (best candidate
per anchor, binned by fit quality and view pair). Shows the bias and spread of the two-view launch point."""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
if len(sys.argv) > 1:
    C.PAIRS_DIR = sys.argv[1]
import eval_mv as EM  # noqa: E402

name, items = EM.hidden_set()
rows = []
for it in items:
    a = it["anchor"]
    if a is None:
        continue
    c = C.best_candidate(a, {"sigma": {"main": 8.0, "end": 4.0}})
    if c is None or c["launch"] is None:
        continue
    xy = it["cands"].get(it["truth"])
    if xy is None:
        continue
    L = c["launch"]
    d = np.asarray(L["xy"]) - np.asarray(xy)
    others = [np.hypot(*(np.asarray(v) - L["xy"])) for t, v in it["cands"].items() if t != it["truth"] and v is not None]
    norm = max(r / (8.0 if v == "main" else 4.0) for r, v in zip(c["rms"], [a["view"], c["vb"]]))
    rows.append((it["view"], c["vb"], norm, d[0], d[1], min(others) if others else np.nan,
                 float(np.sqrt(np.trace(L["cov"]))), L["s"]))
print(len(rows), "judged launches with a fit")
by = defaultdict(list)
for r in rows:
    by[(r[0], r[1], "good" if r[2] <= 2 else ("mid" if r[2] <= 4 else "poor"))].append(r)
print("anchor view / partner / fit   n   median |d| to shooter   to nearest other cand   mean dx dy   sd dx dy   launch sd   lead s")
for k in sorted(by):
    R = np.array([r[2:] for r in by[k]], float)
    dd = np.hypot(R[:, 1], R[:, 2])
    print(f"{k[0]:9s} {k[1]:9s} {k[2]:5s} {len(R):5d}   {np.median(dd):5.2f}   {np.nanmedian(R[:, 3]):5.2f}"
          f"   {R[:, 1].mean():+5.2f} {R[:, 2].mean():+5.2f}   {R[:, 1].std():4.2f} {R[:, 2].std():4.2f}   {np.median(R[:, 4]):4.2f}   {np.median(R[:, 5]):+5.2f}")
