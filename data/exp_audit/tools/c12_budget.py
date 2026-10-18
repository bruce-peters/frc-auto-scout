"""Cycle 12: error budget. Where does the remaining per-team share error live?

Every alliance in output/ is tagged with the known defects found in cycles 3-11:
  no_own_end_cal : its own end camera has no calibration
  cc_bad_main    : Chezy main camera with >= 20 % of main-view boxes off-field (c8_offfield_v2.json)
  v1_detector    : scanned with models/robots.pt (= v1) after the v2 install (new cc matches)
  counter_vs_tba : overlay timeline total off TBA by > 10 %
  identity_contra: a team of the alliance spends >= 15 % of its track time on segments whose own reads
                   clearly name another team
For each tag: alliances, mean total-variation share error vs fuel OPR (and vs scouting), and the share
of the summed squared error it holds. (Tags overlap; 'clean' = no tag.)
"""
import json
from collections import defaultdict

import numpy as np

from common import AUDIT, load, match_keys, opr, scout_counts, tba

off = json.load(open(AUDIT / "c8_offfield_v2.json"))
v1 = set(match_keys("v1"))
sc = scout_counts()
rows = []
for k in match_keys("v2"):
    r = load(k, "v2"); info = tba.match(k); op = opr(info.event_key)
    for al in ("red", "blue"):
        tm = info.teams(al)
        f = np.array([r["teams"][str(t)]["fuel"] for t in tm]); o = np.array([op.get(t, np.nan) for t in tm])
        if f.sum() < 40 or not np.all(np.isfinite(o)) or o.sum() <= 0:
            continue
        tags = set()
        if f"{al}_end" not in r["calibration"]:
            tags.add("no_own_end_cal")
        if k.startswith("2026cc") and k in off:
            pv = off[k]["per_view"].get("main", [1, 0])
            if pv[1] / max(pv[0], 1) >= 0.2:
                tags.add("cc_bad_main")
        if k.startswith("2026cc") and k not in v1:
            tags.add("v1_detector")
        meta = r["teams"][f"_{al}"]
        if meta.get("tba_total") and meta.get("timeline_total") and abs(meta["timeline_total"] / meta["tba_total"] - 1) > 0.10:
            tags.add("counter_vs_tba")
        ide = r["identity"][al]
        for i, t in enumerate(tm):
            segs = [e for e in ide.values() if e["team"] == t]
            dur = sum(e["t1"] - e["t0"] for e in segs) or 1e-9
            contra = sum(e["t1"] - e["t0"] for e in segs if sum(e["ocr"]) >= 1.5 and max(e["ocr"]) / sum(e["ocr"]) >= 0.7
                         and int(np.argmax(e["ocr"])) != i)
            if contra / dur >= 0.15:
                tags.add("identity_contra")
        e_opr = float(np.abs(f / f.sum() - o / o.sum()).sum() / 2)
        e_sc = None
        if k in sc and all(t in sc[k] for t in tm):
            s = np.array([sc[k][t] for t in tm])
            if s.sum() > 0:
                e_sc = float(np.abs(f / f.sum() - s / s.sum()).sum() / 2)
        rows.append({"key": k, "al": al, "tags": sorted(tags), "e_opr": e_opr, "e_sc": e_sc})
tot2 = sum(x["e_opr"] ** 2 for x in rows)
tot2s = sum(x["e_sc"] ** 2 for x in rows if x["e_sc"] is not None)
print(f"{len(rows)} alliances; mean share error (total variation) vs OPR {np.mean([x['e_opr'] for x in rows]):.3f}")
groups = defaultdict(list)
for x in rows:
    for t in x["tags"] or ["clean"]:
        groups[t].append(x)
print(f"   {'tag':16s} {'alliances':>9s} {'err vs OPR':>10s} {'% of sq err':>11s} {'err vs scout':>12s} {'% of sq err':>11s}")
for t, g in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    es = [x["e_sc"] for x in g if x["e_sc"] is not None]
    print(f"   {t:16s} {len(g):9d} {np.mean([x['e_opr'] for x in g]):10.3f} {100 * sum(x['e_opr'] ** 2 for x in g) / tot2:10.1f}% "
          f"{(np.mean(es) if es else float('nan')):12.3f} {(100 * sum(e ** 2 for e in es) / tot2s if es else 0):10.1f}%  (scouted {len(es)})")
json.dump(rows, open(AUDIT / "c12_budget.json", "w"), indent=1)
