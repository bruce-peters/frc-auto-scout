"""Shared helpers for the multi-view experiment: robot tracks from result.json, anchor lookup, candidate choice."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from autoscout import config, triangulate as T  # noqa: E402

EXP = config.DATA / "exp_multiview"
AUTO_LEN = config.AUTO_LEN
PAIRS_DIR = "pairs"  # pass-1 output folder under data/exp_multiview (pairs_lm: landmark main camera)


def load_pairs(k):
    p = EXP / PAIRS_DIR / f"{k}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as fh:
        return pickle.load(fh)


def anchor_index(pairs):
    return {(a["view"], round(a["t0"], 4), round(a["u0"], 1), round(a["v0"], 1)): a for a in pairs["anchors"]}


def anchor_key(view, pts):
    return (view, round(float(pts[0][0]), 4), round(float(pts[0][1]), 1), round(float(pts[0][2]), 1))


def team_tracks(result: dict) -> dict:
    """{team: (n, 3) array of (video time, x, y)} from result.json paths (match time -> video time)."""
    tl = result["timeline"]
    a0, ts = tl.get("auto_start"), tl.get("teleop_start")
    out = {}
    for t, rows in result["paths"].items():
        arr = []
        for mt, x, y in rows:
            vt = (ts + (mt - AUTO_LEN)) if (ts is not None and mt >= AUTO_LEN) else a0 + mt
            arr.append((vt, x, y))
        if len(arr) >= 2:
            a = np.array(sorted(arr))
            out[int(t)] = a
    return out


def position(arr, t, max_gap=0.6, hold=1.0):
    """(x, y, observed) of a team at video time t: the nearest sighting within max_gap s, else linear
    interpolation across a gap of <= 3 s, else the first / last sighting within `hold` s; None otherwise."""
    if arr is None or not len(arr):
        return None
    j = int(np.searchsorted(arr[:, 0], t))
    near = [arr[k] for k in (j - 1, j) if 0 <= k < len(arr) and abs(arr[k][0] - t) <= max_gap]
    if near:
        c = min(near, key=lambda r: abs(r[0] - t))
        if 0 < j < len(arr) and arr[j][0] - arr[j - 1][0] <= 1.0:
            a, b = arr[j - 1], arr[j]
            f = (t - a[0]) / max(b[0] - a[0], 1e-6)
            return float(a[1] + f * (b[1] - a[1])), float(a[2] + f * (b[2] - a[2])), True
        return float(c[1]), float(c[2]), True
    if 0 < j < len(arr):
        a, b = arr[j - 1], arr[j]
        if b[0] - a[0] <= 3.0:
            f = (t - a[0]) / max(b[0] - a[0], 1e-6)
            return float(a[1] + f * (b[1] - a[1])), float(a[2] + f * (b[2] - a[2])), False
        return None
    if j == 0 and arr[0][0] - t <= hold:
        return float(arr[0][1]), float(arr[0][2]), False
    if j == len(arr) and t - arr[-1][0] <= hold:
        return float(arr[-1][1]), float(arr[-1][2]), False
    return None


def by_view_track(pairs):
    return {(a["view"], a["i"]): a for a in pairs["anchors"]}


def _norm(anchor, c, rule):
    sig = rule.get("sigma") or {}
    views = [anchor["view"], c["vb"]]
    return max(r / sig.get("main" if v == "main" else "end", 1.0) for r, v in zip(c["rms"], views))


def mutual_ok(anchor, c, rule, bvt) -> bool:
    """If the partner track is itself a launch anchor, its own best partner in this view must be this anchor."""
    b = bvt.get((c["vb"], c["ib"])) if bvt else None
    if b is None:
        return rule.get("mutual") != "strict"
    cs = [x for x in b["cands"] if x["vb"] == anchor["view"]]
    if not cs:
        return False
    best = min(cs, key=lambda x: _norm(b, x, rule))
    return best["ib"] == anchor["i"]


def best_candidate(anchor, rule: dict, bvt: dict | None = None):
    """The anchor's accepted partner fit under `rule`, or None.

    rule keys: max_rms (px, every view), min_common (instants), sigma (per-view px noise for the normalised cost:
    {"main": .., "end": ..}), ratio (the best normalised cost must beat the next candidate's by this factor),
    need_launch (a launch point is required)."""
    cs = []
    for c in anchor["cands"]:
        if rule.get("need_launch", True) and c["launch"] is None:
            continue
        if c["n_common"] < rule.get("min_common", 3):
            continue
        cs.append((_norm(anchor, c, rule), c))
    if not cs:
        return None
    cs.sort(key=lambda r: r[0])
    norm, c = cs[0]
    if norm > rule.get("max_norm", 1e9):
        return None
    if rule.get("ratio") and len(cs) > 1 and cs[1][0] < rule["ratio"] * norm:
        return None
    if rule.get("mutual") and not mutual_ok(anchor, c, rule, bvt):
        return None
    if rule.get("smooth_partner") and not (c.get("b_res", 0.0) < rule["smooth_partner"]):
        return None
    return c
