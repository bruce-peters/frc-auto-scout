"""How often do two different robot tracks (segments) carry the same team label at the same time? (identity
exclusivity violated: one of them is another robot). Per match, from the in-memory cache dets.json (t, view, box,
alliance, team, seg): share of labelled boxes whose label is also on another segment in the same view and frame, and
the launches tied to such boxes (shot launch point inside / nearest such box at the nearest det frame).

  .venv/Scripts/python data/exp_visual_qa/tools/dup_labels.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from autoscout import views  # noqa: E402

C = ROOT / "data/exp_visual_qa/cache"
tot_b = tot_dup = tot_s = tot_sdup = 0
rows = []
for d in sorted(C.iterdir()):
    if not (d / "dets.json").exists():
        continue
    dets = json.loads((d / "dets.json").read_text())
    res = json.loads((d / "result.json").read_text())
    meta = json.loads((d / "meta.json").read_text())
    off = {v.name: v.offset(1920, 1080) for v in views.PRODUCTIONS[meta["prod"]]}
    fr = defaultdict(list)
    for r in dets:
        if r[7] is not None:
            fr[(r[0], r[1])].append(r)
    nb = nd = 0
    dupf = defaultdict(set)  # (view) -> set of (t, team) duplicated
    for (t, vw), rs in fr.items():
        segs = defaultdict(set)
        for r in rs:
            segs[r[7]].add(r[8])
        for r in rs:
            nb += 1
            if len(segs[r[7]]) > 1:
                nd += 1
                dupf[vw].add((t, r[7]))
    keys = {vw: np.array(sorted({t for (t, v) in fr if v == vw})) for vw in {v for (_, v) in fr}}
    ns = nsd = 0
    for s in res["shots"]:
        if s["how"] != "box" or s["team"] is None:
            continue
        ns += 1
        kk = keys.get(s["view"])
        if kk is None or not len(kk):
            continue
        t = kk[int(np.argmin(np.abs(kk - s["t"])))]
        if (t, s["team"]) in dupf[s["view"]]:
            nsd += 1
    tot_b += nb
    tot_dup += nd
    tot_s += ns
    tot_sdup += nsd
    rows.append((d.name, nb, nd, ns, nsd))
for r in sorted(rows, key=lambda r: -r[4])[:15]:
    print(f"{r[0]:14s} boxes {r[1]:6d} dup {100 * r[2] / max(r[1], 1):5.1f} %   box launches {r[3]:5d} on dup-labelled boxes {r[4]:4d} ({100 * r[4] / max(r[3], 1):.1f} %)")
print(f"ALL: {len(rows)} matches, labelled boxes {tot_b}, duplicate label {100 * tot_dup / tot_b:.1f} %; box launches {tot_s},"
      f" tied to a duplicate-labelled box {tot_sdup} ({100 * tot_sdup / tot_s:.1f} %)")
