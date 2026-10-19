"""Rough scale of launches that never become shots: flight tracks (split, as analyse() sees them) that rise clearly
(>= 6 points, climb >= 40 px from the first point, start going up) but are not in result.json's shots (matched on
view + first point). Upper bound: includes bounces off robots/hub, human-player throws and arcs the gate rejects.
Also reports how many of them start within 2 s of a team's box-linked launch nowhere near (> 200 px).

  .venv/Scripts/python data/exp_visual_qa/tools/unlinked_arcs.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
C = ROOT / "data/exp_visual_qa/cache"
tot_shots = tot_rise = tot_unl = 0
rows = []
for d in sorted(C.iterdir()):
    if not (d / "tracks.pkl").exists():
        continue
    r = json.loads((d / "result.json").read_text())
    with open(d / "tracks.pkl", "rb") as fh:
        tr = pickle.load(fh)
    keys = {(s["view"], round(s["t"], 3), round(s["u"], 1), round(s["v"], 1)) for s in r["shots"]}
    n_rise = n_unl = 0
    for vw, lst in tr.items():
        for p in lst:
            if len(p) < 6:
                continue
            a = np.array(p)
            climb = a[0, 2] - a[:, 2].min()
            if climb < 40 or a[min(3, len(a) - 1), 2] >= a[0, 2]:
                continue
            n_rise += 1
            if (vw, round(a[0, 0], 3), round(a[0, 1], 1), round(a[0, 2], 1)) not in keys:
                n_unl += 1
    tot_shots += len(r["shots"])
    tot_rise += n_rise
    tot_unl += n_unl
    rows.append((d.name, len(r["shots"]), n_rise, n_unl))
print(f"{len(rows)} matches: shots {tot_shots}; clearly rising tracks {tot_rise}, of which not a shot {tot_unl}"
      f" ({100 * tot_unl / max(tot_rise, 1):.0f} %)")
for row in sorted(rows, key=lambda x: -x[3] / max(x[2], 1))[:8]:
    print(row)
