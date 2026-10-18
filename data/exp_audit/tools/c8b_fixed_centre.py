"""Cycle 8b: main camera with a fixed centre (pan/tilt only) instead of a free 6-DOF pose every 4 s.

The window poses fit their tags to ~0.5 px but jump 0.5-2.4 m in centre between windows (4 small tags,
~26 m away: depth/rotation ambiguity). Here: C = median centre of the window poses; for each window,
project the tags that pose sees (tags inside the view) and re-solve only the rotation with C fixed
(LM on those projected corners, so no video is needed). Then, in memory (OCR: cached reads only):
off-field share of main boxes, held-out identity, share corr vs OPR and vs scouting.
Usage: c8b_fixed_centre.py SRC match...   (writes data/exp_audit/c8b_<n>.json)
"""
import gc
import json
import pickle
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

from common import AUDIT, DIRS, P, scout_metrics, share_metrics, tba
from autoscout import field as F, ocr, robots, views

ocr.read_numbers = lambda img: []
src, keys = sys.argv[1], sys.argv[2:]
TAGS = F.load_tags()
CORNERS = {i: F.tag_corners(t) for i, t in TAGS.items()}


def fixed_centre(cam):
    K, D = np.array(cam.K), np.array(cam.dist)
    W, H = cam.size
    cents = np.array([cam.at(t).center_world() for t in cam.times])
    C = np.median(cents, axis=0)
    rv_new, tv_new, errs = [], [], []
    for i, t in enumerate(cam.times):
        c = cam.at(t)
        obj, img = [], []
        for tid, cn in CORNERS.items():
            pc = (c.R @ (cn - c.center_world()).T).T
            if (pc[:, 2] <= 0.5).any():
                continue
            p = c.project(cn)
            if (p[:, 0] >= 0).all() and (p[:, 0] < W).all() and (p[:, 1] >= 0).all() and (p[:, 1] < H).all():
                obj.append(cn); img.append(p)
        if len(obj) < 1:
            rv_new.append(list(cam.rvecs[i])); tv_new.append(list(cam.tvecs[i])); errs.append(np.nan)
            continue
        obj, img = np.concatenate(obj), np.concatenate(img)

        def res(rv):
            R, _ = cv2.Rodrigues(rv)
            p, _ = cv2.projectPoints(obj, rv, -R @ C, K, D)
            return (p.reshape(-1, 2) - img).ravel()
        s = least_squares(res, np.array(cam.rvecs[i], float), method="lm")
        R, _ = cv2.Rodrigues(s.x)
        rv_new.append(s.x.tolist()); tv_new.append((-R @ C).tolist())
        errs.append(float(np.sqrt((s.fun.reshape(-1, 2) ** 2).sum(1).mean())))
    tr = F.CameraTrack(cam.K, cam.dist, list(cam.times), rv_new, tv_new, cam.size, cam.rms, cam.n_tags)
    return tr, C, np.nanmedian(errs), np.nanmax(errs), cents.std(0)


def off_share(md, cam):
    v = next(x for x in views.PRODUCTIONS[md.prod] if x.name == "main")
    ox, oy = v.offset(1920, 1080)
    n = o = 0
    for d in md.dets:
        if d.view != "main":
            continue
        c = cam.at(d.t)
        xy = c.backproject(np.array([[(d.box[0] + d.box[2]) / 2 - ox, d.box[3] - oy]]), z=0.03)[0]
        n += 1
        if not np.all(np.isfinite(xy)):
            o += 1; continue
        ray = xy - c.center_world()[:2]
        xy = xy + P.ROBOT_HALF_DEPTH * ray / max(np.hypot(*ray), 1e-6)
        o += int(not robots.in_field(xy))
    return o / max(n, 1)


pm = {"orig": {}, "fixed": {}}
held = {"orig": [0.0, 0.0], "fixed": [0.0, 0.0]}
rows = []
for k in keys:
    with open(DIRS[src] / k / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    info = md.info
    cam0 = md.cams.get("main")
    if cam0 is None:
        continue
    cam1, C, e_med, e_max, spread = fixed_centre(cam0)
    o0, o1 = off_share(md, cam0), off_share(md, cam1)
    for name, cam in (("orig", cam0), ("fixed", cam1)):
        md.cams["main"] = cam
        res = P.analyse(md, P.Options())
        pm[name][k] = {t: res["teams"][t]["fuel"] for t in info.red + info.blue}
        for hk in (0, 1):
            rh = P.analyse(md, P.Options(ocr_holdout=hk))
            for al in ("red", "blue"):
                tm = info.teams(al)
                for s in rh["identity"][al].values():
                    if s["team"] is not None:
                        held[name][0] += s["held"][tm.index(s["team"])]
                        held[name][1] += sum(s["held"])
    md.cams["main"] = cam0
    rows.append({"key": k, "centre": np.round(C, 2).tolist(), "centre_spread_m": np.round(spread, 2).tolist(),
                 "refit_err_px_med": round(float(e_med), 2), "refit_err_px_max": round(float(e_max), 2),
                 "off_orig": round(o0, 3), "off_fixed": round(o1, 3)})
    print(rows[-1], flush=True)
    del md
    gc.collect()
out = {}
for name in pm:
    out[name] = {"heldout": held[name][0] / max(held[name][1], 1e-9), **share_metrics(pm[name]), **(scout_metrics(pm[name]) or {})}
    print(name, out[name])
json.dump({"rows": rows, "metrics": out, "fuel": pm}, open(AUDIT / f"c8b_{len(keys)}.json", "w"), indent=1, default=float)
