"""Shares of the cycle-16 problem robots (and every robot flagged dead or parked) per variant of a run.

  cases.py RUN [variants...]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from summarize import load_runs, scout_alliances  # noqa: E402

CASES = [("2026cc_qm3", 6036), ("2026cc_qm18", 3045), ("2026cc_qm19", 4499), ("2026cc_qm23", 4499),
         ("2026cc_qm30", 5507), ("2026cc_qm17", 6665), ("2026cc_qm24", 5507), ("2026cc_qm39", 3045)]
run = sys.argv[1]
recs = load_runs([run])
vs = sys.argv[2:] or sorted({v for r in recs.values() for v in r["v"]})
als = {(x["key"], x["al"]): x for x in scout_alliances()}
rows = list(CASES)
for k, r in recs.items():  # every team flagged dead / parked by any variant, in scouted alliances
    for v in r["v"].values():
        for t in v.get("absent_real", []) or []:
            pass
for k, r in sorted(recs.items()):
    for vn, v in r["v"].items():
        res = v.get("flags") or {}
        for t in res.get("dead", []) + [int(tt) for tt, e in (res.get("parked") or {}).items() if e is not None and e < 1.5]:
            if (k, int(t)) not in rows:
                rows.append((k, int(t)))
print(f"{'match':13s} {'team':>5s} {'scout':>5s} " + " ".join(f"{v:>14s}" for v in vs) + "  flags")
for k, t in rows:
    r = recs.get(k)
    if r is None:
        continue
    al = "red" if t in r["red"] else "blue"
    x = als.get((k, al))
    tm = r[al]
    ss = f"{x['scout'][tm.index(t)] / x['scout'].sum():.2f}" if x is not None else "  -"
    cells = []
    for v in vs:
        if v not in r["v"]:
            cells.append(f"{'-':>14s}")
            continue
        f = np.array([r["v"][v]["fuel"][str(u)] for u in tm], float)
        cells.append(f"{f[tm.index(t)] / max(f.sum(), 1e-9):14.2f}")
    fl = []
    for v in vs:
        g = (r["v"].get(v) or {}).get("flags") or {}
        if t in g.get("dead", []):
            fl.append(f"{v}:dead({sum(b - a for a, b in g.get('absent', {}).get(str(t), [])):.0f}s absent)")
        pe = (g.get("parked") or {}).get(str(t))
        if pe is not None and pe < 1.5:
            fl.append(f"{v}:parked({pe})")
    print(f"{k:13s} {t:5d} {ss:>5s} " + " ".join(cells) + "  " + "; ".join(sorted(set(fl))[:3]))
