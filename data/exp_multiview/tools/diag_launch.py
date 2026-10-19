"""Diagnostic: are the cross-view fits real? A real pair's launch point lands on a robot; a chance pair's does not.

For each anchor's best candidate (lowest max-view rms, launch point required): distance from the launch point to
the nearest robot (result.json paths) at the launch time, and the same against the robots 20 s later (null).
Binned by the fit rms (px) and the view pair.

  .venv/Scripts/python data/exp_multiview/tools/diag_launch.py [matches...]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from autoscout import config  # noqa: E402

keys = sys.argv[1:] or sorted(p.stem for p in (C.EXP / "pairs").glob("*.pkl"))
rows = []
for k in keys:
    P = C.load_pairs(k)
    rp = config.OUTPUT / k / "result.json"
    if P is None or not rp.exists():
        continue
    trk = C.team_tracks(json.loads(rp.read_text()))
    for a in P["anchors"]:
        cs = [c for c in a["cands"] if c["launch"] is not None]
        if not cs:
            continue
        c = min(cs, key=lambda c: max(c["rms"]))
        L = c["launch"]
        def dmin(t):
            ds = []
            for tm, arr in trk.items():
                p = C.position(arr, t)
                if p is not None:
                    ds.append(np.hypot(p[0] - L["xy"][0], p[1] - L["xy"][1]))
            return min(ds) if ds else np.inf
        pair = tuple(sorted([a["view"], c["vb"]]))
        rows.append((max(c["rms"]), dmin(L["t"]), dmin(L["t"] + 20.0), pair, c["n_common"], float(np.sqrt(np.trace(L["cov"])))))
R = np.array([(r[0], r[1], r[2], r[4], r[5]) for r in rows])
pairs = [r[3] for r in rows]
print(len(rows), "anchors with a fitted launch point")
bins = [0, 3, 5, 8, 12, 20, 40, 1e9]
for pr in sorted(set(pairs)):
    m = np.array([p == pr for p in pairs])
    print(f"\n{pr}: n={m.sum()}")
    print("   rms bin        n   P(d<0.75) real  null   median d real  null")
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = m & (R[:, 0] >= lo) & (R[:, 0] < hi)
        if s.sum() < 5:
            continue
        print(f"   [{lo:4g},{hi:4g})  {s.sum():6d}   {np.mean(R[s, 1] < 0.75):.3f}   {np.mean(R[s, 2] < 0.75):.3f}"
              f"   {np.median(R[s, 1]):.2f}  {np.median(R[s, 2]):.2f}")
