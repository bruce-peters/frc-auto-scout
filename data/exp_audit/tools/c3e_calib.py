"""Cycle 3e: a missing end-camera calibration. Why does it fail, and what does a fallback buy?

1. tags detected per calibration frame in the uncalibrated view; calibrate_view_dynamic result
2. fallback: intrinsics (+ start pose) from the same view of another match at the event, per-frame
   solvePnP on this match's tags (or the donor's poses if too few tags) -> reprojection error
3. re-run analyse() in memory with the fallback camera; per-team fuel vs scouting (writes nothing
   under output/; results -> data/exp_audit/c3e_<match>.json)
Usage: c3e_calib.py SRC MATCH VIEW DONOR_SRC DONOR_MATCH
"""
import json
import pickle
import sys

import cv2
import numpy as np

from common import AUDIT, DIRS, P, config, load, scout_counts, tba
from autoscout import field as F, views
from autoscout.video import Video

src, key, view, dsrc, dkey = sys.argv[1:6]
import gc

from autoscout import ocr

with open(DIRS[dsrc] / dkey / "scan.pkl", "rb") as fh:
    donor = pickle.load(fh).cams[view]  # keep only the camera
gc.collect()
with open(DIRS[src] / key / "scan.pkl", "rb") as fh:
    md = pickle.load(fh)
print("cnn_cache:", None if getattr(md, "cnn_cache", None) is None else len(md.cnn_cache),
      "ocr_cache:", len(getattr(md, "ocr_cache", {}) or {}), flush=True)
N_SKIP = [0]


def _no_ocr(img):  # memory/CPU: no new bumper reads (identity of new segments from cached reads + appearance)
    N_SKIP[0] += 1
    return []


ocr.read_numbers = _no_ocr
tl = md.timeline
vid = Video(config.VIDEOS / f"{key}.mp4")
v = next(x for x in views.PRODUCTIONS[md.prod] if x.name == view)
tele = tl.teleop_start or (tl.auto_start + P.FIRST_TELEOP_GAP)
times = [float(t) for t in np.arange(tl.auto_start - 1, tele + config.TELEOP_LEN + 2, 4.0)]
frames = [vid.frame_at(t) for t in times]
keep = [(t, f) for t, f in zip(times, frames) if f is not None]
times, crops = [k[0] for k in keep], [v.crop(k[1]) for k in keep]
masks = v.mask_boxes(vid.width, vid.height)
tags = F.load_tags()
ntag = []
seen = {}
for c in crops:
    d = F.detect_tags(c)
    d = {i: q for i, q in d.items() if not any(x0 <= q.mean(0)[0] < x1 and y0 <= q.mean(0)[1] < y1 for x0, y0, x1, y1 in masks)}
    ntag.append(len(d))
    for i in d:
        seen[i] = seen.get(i, 0) + 1
print(f"{key} {view}: {len(crops)} frames, tags per frame {np.bincount(ntag)} (index = #tags); tag ids seen {seen}")
cam = F.calibrate_view_dynamic(crops, times, tags, masks)
print("calibrate_view_dynamic:", None if cam is None else (round(cam.rms, 2), len(cam.times), cam.n_tags))
print("donor", dkey, view, "rms", round(donor.rms, 2), "poses", len(donor.times), "tags", donor.n_tags)

# fallback: donor intrinsics, per-frame PnP from this match's tags where >= 2 tags, else donor mid pose
K, D = np.array(donor.K), np.array(donor.dist)
mid = donor.at(float(np.median(donor.times)))
t_l, rv_l, tv_l, errs = [], [], [], []
for t, c in zip(times, crops):
    d = F.detect_tags(c)
    if len(d) < 1:
        continue
    obj = np.concatenate([F.tag_corners(tags[i]) for i in d]).astype(np.float64)
    img = np.concatenate([d[i] for i in d]).astype(np.float64)
    ok, rv, tv = cv2.solvePnP(obj, img, K, D, np.array(mid.rvec), np.array(mid.tvec), useExtrinsicGuess=True,
                              flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        continue
    p, _ = cv2.projectPoints(obj, rv, tv, K, D)
    e = float(np.sqrt(((p.reshape(-1, 2) - img) ** 2).sum(1).mean()))
    p0, _ = cv2.projectPoints(obj, np.array(mid.rvec), np.array(mid.tvec), K, D)
    e0 = float(np.sqrt(((p0.reshape(-1, 2) - img) ** 2).sum(1).mean()))
    errs.append((t, len(d), round(e, 2), round(e0, 2)))
    if e <= 3.0 and len(d) >= 2:
        t_l.append(t); rv_l.append(rv.ravel().tolist()); tv_l.append(tv.ravel().tolist())
print("per-frame (t, tags, PnP err, donor-pose err):", errs[:12], "...")
if t_l:
    fb = F.CameraTrack(donor.K, donor.dist, t_l, rv_l, tv_l, donor.size, float(np.sqrt(np.mean([e[2] ** 2 for e in errs if e[1] >= 2]))), len(seen))
    how = f"donor intrinsics + {len(t_l)} PnP poses"
else:
    fb = donor
    how = "donor camera as is"
print("fallback:", how)

sc = scout_counts().get(key, {})
info = md.info
base = P.analyse(md, P.Options())
md.cams[view] = fb
fixed = P.analyse(md, P.Options())
print("uncached OCR reads skipped:", N_SKIP[0])
out = {"key": key, "view": view, "tags_per_frame": ntag, "fallback": how, "teams": {}}
for al in ("red", "blue"):
    tl_ = info.teams(al)
    for res, name in ((base, "base"), (fixed, "fixed")):
        f = np.array([res["teams"][t]["fuel"] for t in tl_])
        nl = [sum(1 for s in res["shots"] if s["team"] == t and s["toward"]) for t in tl_]
        print(f"  {al:4s} {name:5s} fuel {np.round(f).astype(int).tolist()}  share {np.round(f / f.sum(), 2).tolist()}  launches {nl}")
        out["teams"][f"{al}_{name}"] = {"fuel": f.tolist(), "launches": nl}
    if all(t in sc for t in tl_):
        s = np.array([sc[t] for t in tl_])
        print(f"  {al:4s} scout share {np.round(s / s.sum(), 2).tolist()}  ({tl_})")
        out["teams"][f"{al}_scout"] = s.tolist()
(AUDIT / f"c3e_{key}_{view}.json").write_text(json.dumps(out, indent=1, default=float))
