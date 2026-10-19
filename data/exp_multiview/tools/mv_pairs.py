"""Pass 1: cross-view partners and joint 3-D fits for every launch track (fuel.is_shot) in every view.

Per match (one scan.pkl at a time, read and closed at once; output/ is not written): every is_shot track of every
view is an anchor. For each other view, the tracks whose rays pass within triangulate.PAIR_GAP_M (1 m, median over
>= 3 common instants) of the anchor's are candidates (MultiView.candidates); each candidate gets the closed-form
joint ballistic fit (MultiView.fit_pair: fit_linear on the anchor's points up to the end of the overlap + 0.1 s and
the partner's points inside the overlap +- 0.1 s) and its launch point. Every candidate is kept (up to
PAIR_MAX_CANDS = 12 per anchor and view) so the evaluation can vary the acceptance rule offline.

  .venv/Scripts/python -u data/exp_multiview/tools/mv_pairs.py [--matches ...]
-> data/exp_multiview/pairs/<match>.pkl: {"anchors": [...], "views": {...}, "n_shots": {...}}
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from autoscout import config, fuel, pipeline, scouting, triangulate as T  # noqa: E402

EXP = config.DATA / "exp_multiview"


def anchors_of(mv: T.MultiView) -> dict:
    out = {}
    for vname, vi in mv.v.items():
        idx = []
        for i, tr in enumerate(vi.tracks):
            cam = vi.cam.at(tr.pts[0][0])
            ppm = pipeline._px_per_m(cam, tr.pts[0][1], tr.pts[0][2])
            if fuel.is_shot(tr, ppm):
                idx.append(i)
        out[vname] = idx
    return out


def fit_candidate(mv, va, i, vb, ib, tc, shot_sets=None):
    """triangulate.MultiView.fit_pair, stored compactly with the partner's own track quality."""
    f = mv.fit_pair(va, i, vb, ib, tc)
    if f is None:
        return None
    trb = mv.v[vb].tracks[ib]
    lp = f["launch"]
    return {"vb": vb, "ib": int(ib), "rms": f["rms"], "n": f["n"], "n_common": f["n_common"], "x": f["x"],
            "b_res": float(trb.fit()["res"]) if len(trb.pts) >= 3 else np.nan, "b_len": len(trb.pts),
            "b_shot": bool(shot_sets is not None and ib in shot_sets.get(vb, ())),
            "t_ref": f["t_ref"], "cov": f["cov"], "min_depth": f["min_depth"],
            "launch": None if lp is None else {k: lp[k] for k in ("t", "s", "xy", "cov", "vz")}}


def process(md) -> dict:
    mv = T.MultiView(md)
    anc = anchors_of(mv)
    shot_sets = {v: set(x) for v, x in anc.items()}
    rows = []
    for va, idx in anc.items():
        for i in idx:
            tr = mv.v[va].tracks[i]
            cands = []
            for vb in mv.v:
                if vb == va:
                    continue
                for g, ib, tc, mid in mv.candidates(va, i, vb):
                    c = fit_candidate(mv, va, i, vb, ib, tc, shot_sets)
                    if c is not None:
                        c["gap"] = float(g)
                        cands.append(c)
            rows.append({"view": va, "i": int(i), "t0": float(tr.pts[0][0]), "u0": float(tr.pts[0][1]),
                         "v0": float(tr.pts[0][2]), "t1": float(tr.pts[-1][0]), "n": len(tr.pts), "cands": cands})
    return {"anchors": rows, "n_shots": {v: len(x) for v, x in anc.items()},
            "views": {v: {"f": vi.f, "size": list(vi.cam.size)} for v, vi in mv.v.items()}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--lm", action="store_true", help="main camera = the landmark camera (lmcam.py); -> pairs_lm/")
    a = ap.parse_args()
    truth = json.loads((config.DATA / "exp_audit" / "c24_truth_cases.json").read_text())
    keys = a.matches or sorted({c["key"] for c in truth} | set(scouting.by_match(scouting.load())))
    keys = [k for k in keys if (config.OUTPUT / k / "scan.pkl").exists()]
    sub = "pairs_lm" if a.lm else "pairs"
    (EXP / sub).mkdir(parents=True, exist_ok=True)
    if a.lm:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import lmcam  # noqa: E402
    print(len(keys), "matches", flush=True)
    tic = time.time()
    for k in keys:
        out = EXP / sub / f"{k}.pkl"
        if out.exists() and not a.force:
            continue
        with open(config.OUTPUT / k / "scan.pkl", "rb") as fh:
            raw = fh.read()
        md = pickle.loads(raw)
        del raw
        if a.lm:
            md.cams["main"] = lmcam.lm_main_track(md.cams["main"])
        res = process(md)
        res["key"] = k
        with open(out, "wb") as fh:
            pickle.dump(res, fh, protocol=pickle.HIGHEST_PROTOCOL)
        n_c = sum(bool(r["cands"]) for r in res["anchors"])
        print(k, res["n_shots"], "anchors with a candidate", n_c, f"{time.time() - tic:.0f}s", flush=True)
        del md, res
