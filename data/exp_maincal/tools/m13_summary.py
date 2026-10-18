"""Aggregate m12_eval per variant and event group: main-view off-field share, held-out identity, cross-view,
share corr vs fuel OPR / top scorer (scripts/evaluate.py definitions), and for the scouted cc quals the human
scouting comparison (scripts/score_scouting.score). Writes data/exp_maincal/summary.json.
Usage: m13_summary.py [variant ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

from common import EXP, ROOT
from autoscout import opr, tba

sys.path.insert(0, str(ROOT / "scripts"))
import score_scouting  # noqa: E402

OUT = EXP / "eval"
rows = {p.stem: json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))}
variants = sys.argv[1:] or ["before", "at_fix", "guard", "static", "static_floor", "event", "manual"]
_opr = {}


def share(per_match):
    sa, so, top, n_al = [], [], 0, 0
    for key, fuel in per_match.items():
        info = tba.match(key)
        if info.event_key not in _opr:
            _opr[info.event_key] = opr.fuel_opr(info.event_key)
        op = _opr[info.event_key]
        for al in ("red", "blue"):
            ts = info.teams(al)
            fa = np.array([fuel[str(t)] for t in ts])
            o = np.array([op.get(t, np.nan) for t in ts])
            if fa.sum() < 40 or not np.all(np.isfinite(o)) or o.sum() <= 0:
                continue
            sa += (fa / fa.sum()).tolist()
            so += (o / o.sum()).tolist()
            top += int(np.argmax(fa) == np.argmax(o))
            n_al += 1
    return (float(np.corrcoef(sa, so)[0, 1]) if len(sa) > 5 else float("nan")), top, n_al


summary = {}
groups = {"cc": [k for k in rows if k.startswith("2026cc")], "other": [k for k in rows if not k.startswith("2026cc")]}
for gname, ks in groups.items():
    print(f"\n== {gname}: {len(ks)} matches")
    print(f"{'variant':13s} {'n':>3s} {'off-field':>9s} {'held-out':>8s} {'xview m':>7s} {'corr OPR':>8s} {'top':>7s}"
          f" {'scout MAE':>9s} {'scout r':>7s} {'top(sc)':>7s} {'burst':>6s}")
    for v in variants:
        have = [k for k in ks if v in rows[k]["variants"]]
        if not have:
            continue
        R = [rows[k]["variants"][v] for k in have]
        off = float(np.mean([r["off"] for r in R]))
        held = sum(r["agree"] for r in R) / max(1e-9, sum(r["mass"] for r in R))
        xv = float(np.mean([np.mean([x[0] for x in r["xview"].values()]) for r in R if r["xview"]]))
        corr, top, n_al = share({k: rows[k]["variants"][v]["fuel"] for k in have})
        sc = score_scouting.score(OUT / v) if gname == "cc" and (OUT / v).exists() else {}
        summary[f"{gname}/{v}"] = {"n": len(have), "off": off, "heldout": held, "xview": xv, "corr": corr, "top": top,
                                   "alliances": n_al, "scouting": sc}
        s = ""
        if sc:
            s = (f" {sc['video']['mae']:9.1f} {sc['video']['corr']:7.3f} {100 * sc['top_scorer']['video']:6.0f}%"
                 f" {100 * sc.get('burst_overlap', {}).get('video', float('nan')):5.1f}%")
        print(f"{v:13s} {len(have):3d} {off:9.4f} {100 * held:7.1f}% {xv:7.3f} {corr:8.3f} {top:3d}/{n_al:<3d}{s}")
    # per match off-field / held-out for the cc table
    if gname == "cc":
        print("\nper match: off-field before -> at_fix -> static | held-out before -> at_fix -> static")
        for k in sorted(ks):
            V = rows[k]["variants"]

            def g(v, f):
                return f(V[v]) if v in V else float("nan")
            print(f"  {k:16s} " + " -> ".join(f"{g(v, lambda r: r['off']):.3f}" for v in ("before", "at_fix", "static")) + " | " +
                  " -> ".join(f"{g(v, lambda r: r['agree'] / max(r['mass'], 1e-9)):.3f}" for v in ("before", "at_fix", "static")))
(EXP / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
