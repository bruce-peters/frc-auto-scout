"""Cycle 20: re-verify the adopted changes on fresh output/ after the main session's finalize.

Compares the current output/ result.json with a snapshot (argv[1], default snapshot_pre_finalize.json) on the
same matches: share metrics vs scouting (primary) and fuel OPR, with paired bootstrap; plus spot checks:
  counter filter    : alliances whose overlay timeline total is > 10 % off TBA (cc_qm12 blue)
  calibration       : matches missing an end view
  AUTO_ZONE_GATE    : auto share corr vs auto-only OPR (c6b) - was 0.205 gated, 0.394 ungated
  SEG_MAX           : label flips at piece boundaries (c20b logic)
  groups            : original vs new cc share error vs scouting; 5026 share gap vs scouting
"""
import json
import os

import numpy as np

from c6b_auto_opr import period_opr
from common import AUDIT, DIRS, load, match_keys, opr, scout_counts, scout_metrics, share_metrics, tba

import sys
snap = json.load(open(AUDIT / (sys.argv[1] if len(sys.argv) > 1 else "snapshot_pre_finalize.json")))
T0 = snap["taken"]
keys = [k for k in match_keys("v2") if k in snap["matches"]]
fresh = [k for k in keys if os.path.getmtime(DIRS["v2"] / k / "result.json") > T0]
print(f"{len(keys)} matches in both; rewritten since the snapshot: {len(fresh)}")
before = {k: {int(t): f for t, f in snap["matches"][k]["fuel"].items()} for k in fresh}
after = {}
R = {}
for k in fresh:
    R[k] = load(k, "v2")
    after[k] = {int(t): R[k]["teams"][str(t)]["fuel"] for t in tba.match(k).red + tba.match(k).blue}
print("before", share_metrics(before), scout_metrics(before))
print("after ", share_metrics(after), scout_metrics(after))

# paired bootstrap of scout share error (TV per alliance) and OPR corr
sc = scout_counts()


def tv_rows(pm):
    out = {}
    for k, f in pm.items():
        if k not in sc:
            continue
        info = tba.match(k)
        for al in ("red", "blue"):
            tm = info.teams(al)
            if all(t in sc[k] for t in tm):
                v = np.array([f[t] for t in tm]); s = np.array([sc[k][t] for t in tm])
                if v.sum() > 0 and s.sum() > 0:
                    out[(k, al)] = float(np.abs(v / v.sum() - s / s.sum()).sum() / 2)
    return out


tb, ta = tv_rows(before), tv_rows(after)
common_ = sorted(set(tb) & set(ta))
d = np.array([ta[x] - tb[x] for x in common_])
rng = np.random.default_rng(0)
bs = [rng.choice(d, len(d)).mean() for _ in range(5000)]
print(f"scout share error (TV) after - before: {d.mean():+.4f} over {len(d)} alliances, 95 % CI "
      f"[{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}]")

# spot checks
off = [(k, al, R[k]["teams"][f"_{al}"]["timeline_total"], R[k]["teams"][f"_{al}"]["tba_total"]) for k in fresh for al in ("red", "blue")
       if R[k]["teams"][f"_{al}"].get("tba_total") and R[k]["teams"][f"_{al}"].get("timeline_total")
       and abs(R[k]["teams"][f"_{al}"]["timeline_total"] / R[k]["teams"][f"_{al}"]["tba_total"] - 1) > 0.10]
print("counter filter: alliances > 10 % off TBA:", off)
print("calibration: matches missing a view:", [(k, sorted(R[k]["calibration"])) for k in fresh if len(R[k]["calibration"]) < 3])
OPA, va, vo = {}, [], []
for k in fresh:
    info = tba.match(k)
    if info.event_key not in OPA:
        OPA[info.event_key] = period_opr(info.event_key, lambda h: h.get("autoCount", 0))
    for al in ("red", "blue"):
        tm = info.teams(al)
        v = np.array([R[k]["teams"][str(t)]["fuel_by_period"].get("auto", 0.0) for t in tm])
        o = np.array([OPA[info.event_key].get(t, np.nan) for t in tm])
        if v.sum() >= 15 and np.all(np.isfinite(o)) and o.sum() > 0:
            va += (v / v.sum()).tolist(); vo += (o / o.sum()).tolist()
print(f"auto share corr vs auto-OPR: {np.corrcoef(va, vo)[0, 1]:.3f} (n={len(va) // 3} alliances)")
flips = noread = 0
for k in fresh:
    for al in ("red", "blue"):
        segs = sorted(((e["slot"], e["t0"], e["t1"], e["team"], sum(e["ocr"])) for e in R[k]["identity"][al].values()))
        for a, b in zip(segs[:-1], segs[1:]):
            da, db = a[2] - a[1], b[2] - b[1]
            if a[0] == b[0] and 0 <= b[1] - a[2] < 0.25 and 5 <= da <= 10.2 and 5 <= db <= 10.2 and abs(da - db) < 0.4 and b[4] <= 0.5:
                noread += 1; flips += int(a[3] != b[3])
print(f"SEG_MAX: unsupported flips at read-less piece boundaries {flips}/{noread} ({100 * flips / max(noread, 1):.1f} %)")
ORIG = set(match_keys("v1"))
for name, sel in (("original cc", lambda k: k in ORIG), ("new cc", lambda k: k not in ORIG)):
    e_b = [v for (k, al), v in tb.items() if k.startswith("2026cc") and sel(k)]
    e_a = [v for (k, al), v in ta.items() if k.startswith("2026cc") and sel(k)]
    print(f"{name}: scout share error before {np.mean(e_b) if e_b else float('nan'):.3f} -> after {np.mean(e_a) if e_a else float('nan'):.3f} (n={len(e_a)})")
for name, pm in (("before", before), ("after", after)):
    g = []
    for k, f in pm.items():
        if k in sc and 5026 in sc[k]:
            info = tba.match(k); al = "red" if 5026 in info.red else "blue"; tm = info.teams(al)
            if all(t in sc[k] for t in tm):
                v = np.array([f[t] for t in tm]); s = np.array([sc[k][t] for t in tm])
                g.append(v[tm.index(5026)] / v.sum() - s[tm.index(5026)] / s.sum())
    print(f"5026 share gap vs scouting {name}: {np.mean(g):+.3f} (n={len(g)})")
