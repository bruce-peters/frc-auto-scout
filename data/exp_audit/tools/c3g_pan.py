"""Cycle 3g: pan-only fallback for an end camera that saw too few tags to calibrate.

End cameras sit on fixed mounts; between matches the operator pans/tilts them. So take the intrinsics
and the camera centre from the same view of another match at the event, and solve only the rotation
per sampled frame from whatever tags are visible (1 tag = 4 corners = 8 equations for 3 unknowns).
Validation without scouting: back-project the view's robot boxes and compare with the same alliance's
fused positions from the other views (nearest-robot distance), next to the donor match's own numbers.
Then analyse() in memory -> per-team fuel vs scouting. Writes data/exp_audit/c3g_<match>.json.
Usage: c3g_pan.py SRC MATCH VIEW DONOR_MATCH
"""
import gc
import json
import pickle
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

from common import AUDIT, DIRS, P, config, scout_counts
from autoscout import field as F, ocr, views
from autoscout.video import Video

src, key, view, dkey = sys.argv[1:5]
with open(DIRS[src] / dkey / "scan.pkl", "rb") as fh:
    dmd = pickle.load(fh)
donor = dmd.cams[view]
# donor-match cross-view agreement baseline (computed below with the same function)
d_dets = [d for d in dmd.dets]
d_cams = dict(dmd.cams)
d_prod = dmd.prod
del dmd
gc.collect()
K, D = np.array(donor.K), np.array(donor.dist)
centres = np.array([donor.at(t).center_world() for t in donor.times])
C = np.median(centres, axis=0)
print(f"donor {dkey} {view}: rms {donor.rms:.2f}, camera centre {np.round(C, 2)} (spread {np.round(centres.std(0), 3)} m)")

with open(DIRS[src] / key / "scan.pkl", "rb") as fh:
    md = pickle.load(fh)
ocr.read_numbers = lambda img: []  # no new bumper reads (memory / CPU)
tl = md.timeline
vid = Video(config.VIDEOS / f"{key}.mp4")
v = next(x for x in views.PRODUCTIONS[md.prod] if x.name == view)
tele = tl.teleop_start or (tl.auto_start + P.FIRST_TELEOP_GAP)
times = [float(t) for t in np.arange(tl.auto_start - 1, tele + config.TELEOP_LEN + 2, 4.0)]
tags = F.load_tags()


def solve_rot(obj, img, r0):
    def res(rv):
        R, _ = cv2.Rodrigues(rv)
        tv = -R @ C
        p, _ = cv2.projectPoints(obj, rv, tv, K, D)
        return (p.reshape(-1, 2) - img).ravel()
    best = None
    for dy in (0.0, 0.15, -0.15, 0.3, -0.3):
        r_init = cv2.Rodrigues(cv2.Rodrigues(np.array(r0, float))[0] @ cv2.Rodrigues(np.array([0, dy, 0.0]))[0])[0].ravel()
        s = least_squares(res, r_init, method="lm")
        e = float(np.sqrt((s.fun.reshape(-1, 2) ** 2).sum(1).mean()))
        if best is None or e < best[1]:
            best = (s.x, e)
    return best


r0 = donor.at(float(np.median(donor.times))).rvec
t_l, rv_l, tv_l, errs = [], [], [], []
for t in times:
    f = vid.frame_at(t)
    if f is None:
        continue
    det = F.detect_tags(v.crop(f))
    if not det:
        continue
    obj = np.concatenate([F.tag_corners(tags[i]) for i in det]).astype(np.float64)
    img = np.concatenate([det[i] for i in det]).astype(np.float64)
    rv, e = solve_rot(obj, img, r0)
    errs.append((round(t, 1), len(det), round(e, 2)))
    if e <= 3.0:
        R, _ = cv2.Rodrigues(rv)
        t_l.append(t); rv_l.append(rv.tolist()); tv_l.append((-R @ C).tolist())
        r0 = rv
print("per-frame (t, tags, err px):", errs)
if len(t_l) < 2:
    sys.exit("too few poses")
fb = F.CameraTrack(donor.K, donor.dist, t_l, rv_l, tv_l, donor.size,
                   float(np.sqrt(np.mean([e[2] ** 2 for e in errs if e[2] <= 3.0]))), len({1}))
yaw = [np.degrees(np.linalg.norm(cv2.Rodrigues(cv2.Rodrigues(np.array(a))[0].T @ cv2.Rodrigues(np.array(b))[0])[0]))
       for a, b in zip(rv_l[:-1], rv_l[1:])]
print(f"fallback: {len(t_l)} poses, rms {fb.rms:.2f} px, pose-to-pose rotation median {np.median(yaw):.2f} deg max {max(yaw):.2f}")


def cross_view(dets, cams, prod, vname):
    """median / q75 distance (m) from each vname box's floor point to the nearest same-alliance box floor
    point from another view within 0.05 s (both back-projected like analyse, + ROBOT_HALF_DEPTH)."""
    vl = {x.name: x for x in views.PRODUCTIONS[prod]}
    def xy(d):
        cam = cams.get(d.view)
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
    by_t = {}
    for d in dets:
        by_t.setdefault(round(d.t, 2), []).append(d)
    out = []
    for t, ds in by_t.items():
        a = [(d, xy(d)) for d in ds if d.view == vname]
        b = [(d, xy(d)) for d in ds if d.view != vname]
        for d, p in a:
            if p is None:
                continue
            q = [np.hypot(*(p - pp)) for dd, pp in b if pp is not None and dd.alliance == d.alliance]
            if q:
                out.append(min(q))
    out = np.array(out)
    return (float(np.median(out)), float(np.quantile(out, .75)), len(out)) if len(out) else None


print("cross-view agreement (median, q75 m, n): donor match", cross_view(d_dets, d_cams, d_prod, view))
del d_dets, d_cams
gc.collect()
cams2 = dict(md.cams); cams2[view] = fb
print("                                       fallback     ", cross_view(md.dets, cams2, md.prod, view))
sc = scout_counts().get(key, {})
base = P.analyse(md, P.Options())
md.cams[view] = fb
fixed = P.analyse(md, P.Options())
out = {"key": key, "view": view, "poses": len(t_l), "rms": fb.rms, "teams": {}}
for al in ("red", "blue"):
    tm = md.info.teams(al)
    for res, name in ((base, "base"), (fixed, "pan-fallback")):
        f = np.array([res["teams"][t]["fuel"] for t in tm])
        nl = [sum(1 for s in res["shots"] if s["team"] == t and s["toward"]) for t in tm]
        print(f"  {al:4s} {name:12s} fuel {np.round(f).astype(int).tolist()} share {np.round(f / f.sum(), 2).tolist()} launches {nl}")
        out["teams"][f"{al}_{name}"] = {"fuel": f.tolist(), "launches": nl}
    if all(t in sc for t in tm):
        s = np.array([sc[t] for t in tm])
        print(f"  {al:4s} scout share  {np.round(s / s.sum(), 2).tolist()}  scouted shots {s.round().tolist()} {tm}")
(AUDIT / f"c3g_{key}_{view}.json").write_text(json.dumps(out, indent=1, default=float))
