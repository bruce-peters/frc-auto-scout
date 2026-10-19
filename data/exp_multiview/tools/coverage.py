"""Coverage: the share of launches that have a two-view partner, per view and view pair.

Launch tracks = pass-1 anchors (fuel.is_shot tracks, every view). Credited launches = output/<match>/result.json
shots (today's pipeline: box or projection link, after the launch gate), matched to anchors by view and first
point. Scored balls: each overlay counter jump of an alliance is covered if a credited launch of that alliance
with an accepted partner lies 0.8-5.5 s before it (the credit window of attribution.split_increments)."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import eval_mv as EM  # noqa: E402
from autoscout import config, scouting  # noqa: E402

if len(sys.argv) > 1:
    C.PAIRS_DIR = sys.argv[1]
rules = {k: EM.RULES[k] for k in ("any fit", "rms<=10", "rms<=6", "norm<=2 (main 8, end 4 px)", "any, mutual strict")}
keys = sorted(p.stem for p in (C.EXP / C.PAIRS_DIR).glob("*.pkl"))
scouted = set(scouting.by_match(scouting.load()))
tot = {rn: {"anchors": Counter(), "anch_pair": Counter(), "cred": Counter(), "cred_pair": Counter(),
            "balls": 0, "balls_cov": 0, "jumps": 0, "jumps_cov": 0} for rn in rules}
n_match = 0
for k in keys:
    P = C.load_pairs(k)
    rp = config.OUTPUT / k / "result.json"
    if P is None or not rp.exists():
        continue
    n_match += 1
    res = json.loads(rp.read_text())
    ix = C.anchor_index(P)
    bvt = C.by_view_track(P)
    acc = {rn: {} for rn in rules}
    for a in P["anchors"]:
        for rn, rule in rules.items():
            c = C.best_candidate(a, rule, bvt)
            acc[rn][(a["view"], round(a["t0"], 4), round(a["u0"], 1), round(a["v0"], 1))] = c
            tot[rn]["anchors"][a["view"]] += 1
            if c is not None:
                tot[rn]["anch_pair"][(a["view"], c["vb"])] += 1
    n_unmatched = 0
    for s in res["shots"]:
        n_unmatched += C.anchor_key(s["view"], [(s["t"], s["u"], s["v"])]) not in ix
        key = C.anchor_key(s["view"], [(s["t"], s["u"], s["v"])])
        for rn in rules:
            c = acc[rn].get(key)
            tot[rn]["cred"][s["view"]] += 1
            if c is not None:
                tot[rn]["cred_pair"][(s["view"], c["vb"])] += 1
    if n_unmatched:
        print(k, "credited launches without an anchor:", n_unmatched)
    for al in ("red", "blue"):
        sh = [s for s in res["shots"] if s["alliance"] == al]
        prev = 0
        for vt, cnt in res["timeline"]["counts"].get(al, []):
            if cnt <= prev:
                continue
            n, prev = cnt - prev, cnt
            win = [s for s in sh if 0.8 <= vt - s["t"] <= 5.5]
            for rn in rules:
                tot[rn]["jumps"] += 1
                tot[rn]["balls"] += n
                if any(acc[rn].get(C.anchor_key(s["view"], [(s["t"], s["u"], s["v"])])) is not None for s in win):
                    tot[rn]["jumps_cov"] += 1
                    tot[rn]["balls_cov"] += n
out = {"matches": n_match}
for rn, t in tot.items():
    A = sum(t["anchors"].values())
    Cr = sum(t["cred"].values())
    print(f"\n{rn}  ({n_match} matches)")
    print(f"   launch tracks with a partner: {sum(t['anch_pair'].values())} / {A} = {100 * sum(t['anch_pair'].values()) / A:.1f} %")
    for v in ("main", "blue_end", "red_end"):
        nv = t["anchors"][v]
        parts = ", ".join(f"+{vb} {100 * t['anch_pair'][(v, vb)] / max(nv, 1):.1f} %" for vb in ("main", "blue_end", "red_end") if vb != v)
        print(f"     {v:9s} {nv:6d} launch tracks: {parts}")
    print(f"   credited launches with a partner: {sum(t['cred_pair'].values())} / {Cr} = {100 * sum(t['cred_pair'].values()) / max(Cr, 1):.1f} %")
    for v in ("main", "blue_end", "red_end"):
        nv = t["cred"][v]
        parts = ", ".join(f"+{vb} {100 * t['cred_pair'][(v, vb)] / max(nv, 1):.1f} %" for vb in ("main", "blue_end", "red_end") if vb != v)
        print(f"     {v:9s} {nv:6d} credited: {parts}")
    print(f"   scored balls with a two-view credited launch in their credit window: {100 * t['balls_cov'] / max(t['balls'], 1):.1f} % "
          f"({100 * t['jumps_cov'] / max(t['jumps'], 1):.1f} % of counter jumps)")
    out[rn] = {"anchors": {str(k): v for k, v in t["anchors"].items()}, "anch_pair": {"+".join(k): v for k, v in t["anch_pair"].items()},
               "cred": {str(k): v for k, v in t["cred"].items()}, "cred_pair": {"+".join(k): v for k, v in t["cred_pair"].items()},
               "balls": t["balls"], "balls_cov": t["balls_cov"], "jumps": t["jumps"], "jumps_cov": t["jumps_cov"]}
json.dump(out, open(C.EXP / ("coverage.json" if C.PAIRS_DIR == "pairs" else f"coverage_{C.PAIRS_DIR}.json"), "w"), indent=1)
