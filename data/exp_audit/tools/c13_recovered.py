"""Cycle 13: are the recovered end-camera calibrations (field.recover_view / event prior) right?

For each match whose end view was missing in the v1 backup but is calibrated now: cross-view agreement
of that view's robot floor points with same-alliance main-view floor points at the same detection time
(median / q75 nearest distance, m), next to the same view in normally calibrated matches of that event.
Plus the alliance's share error vs OPR before (v1 backup) and now.
"""
import gc
import pickle
from collections import defaultdict

import numpy as np

from common import DIRS, P, load, match_keys, opr, tba
from autoscout import views

CASES = [("2026cagle_qm32", "red_end"), ("2026cagle_sf13m1", "red_end"), ("2026cancmp_qm114", "blue_end"),
         ("2026cancmp_qm52", "blue_end"), ("2026casnv_qm7", "blue_end"), ("2026cc_qm15", "blue_end"), ("2026joh_qm14", "blue_end")]
REF = {"2026cagle": ["2026cagle_qm23", "2026cagle_qm46"], "2026cancmp": ["2026cancmp_qm28", "2026cancmp_qm5"],
       "2026casnv": ["2026casnv_qm2", "2026casnv_qm17"], "2026cc": ["2026cc_qm26", "2026cc_qm30"], "2026joh": ["2026joh_qm3", "2026joh_qm30"]}


def xy_of(md, d, vl):
    cam = md.cams.get(d.view)
    if cam is None:
        return None
    c = cam.at(d.t)
    ox, oy = vl[d.view].offset(1920, 1080)
    p = c.backproject(np.array([[(d.box[0] + d.box[2]) / 2 - ox, d.box[3] - oy]]), z=0.03)[0]
    if not np.all(np.isfinite(p)):
        return None
    ray = p - c.center_world()[:2]
    rng = np.hypot(*ray)
    if d.view != "main" and rng > P.END_VIEW_RANGE_COV:
        return None
    return p + P.ROBOT_HALF_DEPTH * ray / max(rng, 1e-6)


def agreement(key, view):
    with open(DIRS["v2"] / key / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    if view not in md.cams:
        del md; gc.collect()
        return None
    vl = {v.name: v for v in views.PRODUCTIONS[md.prod]}
    by_t = defaultdict(list)
    for d in md.dets:
        if d.view in (view, "main"):
            by_t[round(d.t, 2)].append(d)
    dist = []
    for t, ds in by_t.items():
        a = [(d, xy_of(md, d, vl)) for d in ds if d.view == view]
        b = [(d, xy_of(md, d, vl)) for d in ds if d.view == "main"]
        for d, p in a:
            if p is None:
                continue
            q = [np.hypot(*(p - pp)) for dd, pp in b if pp is not None and dd.alliance == d.alliance]
            if q:
                dist.append(min(q))
    del md; gc.collect()
    dist = np.array(dist)
    return (round(float(np.median(dist)), 2), round(float(np.quantile(dist, .75)), 2), len(dist)) if len(dist) else None


for key, view in CASES:
    ev = key.split("_")[0]
    al = view.split("_")[0]
    info = tba.match(key); op = opr(info.event_key); tm = info.teams(al)
    o = np.array([op[t] for t in tm]); o = o / o.sum()
    errs = {}
    for src in ("v1", "v2"):
        r = load(key, src)
        f = np.array([r["teams"][str(t)]["fuel"] for t in tm])
        errs[src] = round(float(np.abs(f / f.sum() - o).sum() / 2), 3)
    r = load(key, "v2")
    cal = r["calibration"].get(view)
    ag = agreement(key, view)
    refs = [agreement(k, view) for k in REF[ev]]
    print(f"{key:18s} {view:9s} calib now {None if cal is None else (round(cal['rms'], 2), cal['n_tags'])}  cross-view (median, q75 m, n) {ag}"
          f"  | same view in {REF[ev]}: {refs}  | share err vs OPR v1 {errs['v1']} -> now {errs['v2']}", flush=True)
