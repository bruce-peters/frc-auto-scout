"""Pass 2: pick accuracy of the two-view launch point on the scout-judged truth sets of data/exp_arcs.

Sets (all built by data/exp_arcs/tools, matched to pass-1 anchors by view and first track point):
  c24     data/exp_audit/c24_truth_cases.json via eval_truth.c24_cases (1,084 contested launches; today 71.4 %)
  fresh   the c24 construction on the arc-log labels (eval_truth.fresh_cases)
  hidden  every box-linked launch with its boxes and its box-less teammates (eval_hidden.cases_from)

For a case with an accepted two-view fit (common.best_candidate under a rule), the pick is the candidate robot
whose fused position (xy_fused, else the box's own xy_det) is nearest the launch point in Mahalanobis distance
(launch covariance + SIGMA_ROBOT^2), subject to the gate and margin of triangulate.pick_robot. Otherwise (no pair,
or gated) today's pick stands. Paired bootstrap over matches (eval_truth.boot_diff).

  .venv/Scripts/python data/exp_multiview/tools/eval_mv.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "data" / "exp_arcs" / "tools"))
import common as C  # noqa: E402
import eval_truth as E  # noqa: E402
from autoscout import config, triangulate as T  # noqa: E402

RULES = {
    "any fit": {},
    "rms<=15": {"sigma": {"main": 1.0, "end": 1.0}, "max_norm": 15},
    "rms<=10": {"sigma": {"main": 1.0, "end": 1.0}, "max_norm": 10},
    "rms<=6": {"sigma": {"main": 1.0, "end": 1.0}, "max_norm": 6},
    "norm<=2 (main 8, end 4 px)": {"sigma": {"main": 8.0, "end": 4.0}, "max_norm": 2.0},
    "norm<=1.5, ratio 1.5": {"sigma": {"main": 8.0, "end": 4.0}, "max_norm": 1.5, "ratio": 1.5},
    "norm<=2, >=5 common": {"sigma": {"main": 8.0, "end": 4.0}, "max_norm": 2.0, "min_common": 5},
    "norm<=2, mutual": {"sigma": {"main": 8.0, "end": 4.0}, "max_norm": 2.0, "mutual": True},
    "norm<=2, mutual strict": {"sigma": {"main": 8.0, "end": 4.0}, "max_norm": 2.0, "mutual": "strict"},
    "any, mutual strict": {"sigma": {"main": 8.0, "end": 4.0}, "mutual": "strict"},
}
GATES = {"gate": dict(gate=T.GATE_CHI2, margin=T.MARGIN_CHI2), "nearest": dict(gate=1e9, margin=0.0)}

_PAIRS: dict = {}


def _load(key):
    if key not in _PAIRS:
        P = C.load_pairs(key)
        _PAIRS[key] = (C.anchor_index(P), C.by_view_track(P)) if P is not None else (None, None)
    return _PAIRS[key]


def anchor_for(key, view, pts):
    ix = _load(key)[0]
    return None if ix is None else ix.get(C.anchor_key(view, pts))


def mv_pick(launch, cands: dict, gate: dict):
    """cands: {team: (x, y) or None}. The team the launch point picks (pick_robot), or None."""
    rob = {t: xy for t, xy in cands.items() if xy is not None}
    if len(rob) < 1:
        return None, None
    r = T.pick_robot(launch, rob, **gate)
    return r["team"], r


def c24_like(cases, name):
    """cases: [(case, event, idx)] with case['truth'] / case['cross'] (eval_truth construction)."""
    items = []
    for c, e, idx in cases:
        cur = E.pick(e, idx, "current")
        xy = {t: (e["cands"][j]["xy_fused"] or e["cands"][j]["xy_det"]) for t, j in idx.items()}
        items.append({"key": c["key"], "truth": c["truth"], "cross": c["cross"], "today": cur, "view": e["view"],
                      "anchor": anchor_for(c["key"], e["view"], e["pts"]), "cands": xy, "dur": float(e["pts"][-1, 0] - e["pts"][0, 0])})
    return name, items


def hidden_set():
    sys.path.insert(0, str(ROOT / "data" / "exp_arcs" / "tools"))
    import eval_hidden as EH  # noqa: E402
    items = []
    for c in EH.cases_from("log_hidden"):
        e = c["e"]
        xy = {}
        for t, i in list(c["boxes"].items()) + list(c["hid"].items()):
            r = e["cands"][i]
            xy[t] = r["xy_fused"] or r.get("xy_det")
        items.append({"key": c["key"], "truth": c["truth"], "cross": False, "today": c["winner"], "view": c["view"],
                      "anchor": anchor_for(c["key"], e["view"], e["pts"]), "cands": xy, "boxes": set(c["boxes"]),
                      "beh": set(c["beh"]), "hid": set(c["hid"]), "dur": float(e["pts"][-1, 0] - e["pts"][0, 0])})
    return "hidden (log_hidden: boxes + box-less teammates)", items


def evaluate(name, items, rule, gate, boxes_only=False):
    keys, ok_today, ok_new, covered, picks = [], [], [], [], []
    for it in items:
        a = it["anchor"]
        p = None
        if a is not None:
            c = C.best_candidate(a, rule, _load(it["key"])[1])
            if c is not None and c["launch"] is not None:
                cands = it["cands"]
                if boxes_only and "boxes" in it:
                    cands = {t: v for t, v in cands.items() if t in it["boxes"]}
                p, _ = mv_pick(c["launch"], cands, gate)
        covered.append(p is not None)
        new = p if p is not None else it["today"]
        keys.append(it["key"])
        ok_today.append(float(it["today"] == it["truth"]))
        ok_new.append(float(new == it["truth"]))
        picks.append(p)
    return np.array(keys), np.array(ok_today), np.array(ok_new), np.array(covered), picks


def report(name, items, out):
    n = len(items)
    has = sum(it["anchor"] is not None for it in items)
    anyc = sum(bool(it["anchor"] and it["anchor"]["cands"]) for it in items)
    print(f"\n=== {name}: {n} cases in {len({it['key'] for it in items})} matches; anchor found {has}, "
          f"with any cross-view candidate {anyc}")
    print(f"   {'rule':30s} {'gate':8s} {'cover':>6s} {'mv acc':>7s} {'today':>6s}   all: today -> new  diff [95 % CI]")
    res = {}
    for rn, rule in RULES.items():
        for gn, gate in GATES.items():
            keys, ot, on, cov, picks = evaluate(name, items, rule, gate)
            if cov.sum() == 0:
                continue
            lo, hi = E.boot_diff(list(keys), on, ot)
            print(f"   {rn:30s} {gn:8s} {100 * cov.mean():5.1f}% {100 * on[cov].mean():6.1f}% {100 * ot[cov].mean():5.1f}%"
                  f"   {100 * ot.mean():5.1f} -> {100 * on.mean():5.1f}  {100 * (on.mean() - ot.mean()):+5.1f} [{100 * lo:+5.1f}, {100 * hi:+5.1f}]")
            res[f"{rn} | {gn}"] = {"cover": float(cov.mean()), "n_cover": int(cov.sum()), "mv_acc": float(on[cov].mean()),
                                   "today_acc_on_cover": float(ot[cov].mean()), "today": float(ot.mean()),
                                   "new": float(on.mean()), "ci": [float(lo), float(hi)]}
    out[name] = res


if __name__ == "__main__":
    if len(sys.argv) > 1:
        C.PAIRS_DIR = sys.argv[1]
    out = {}
    truth, matched = E.c24_cases()
    for name, items in (c24_like(matched, "c24"), c24_like(E.fresh_cases(), "fresh"), hidden_set()):
        report(name, items, out)
    json.dump(out, open(C.EXP / ("eval_mv.json" if C.PAIRS_DIR == "pairs" else f"eval_mv_{C.PAIRS_DIR}.json"), "w"), indent=1)
