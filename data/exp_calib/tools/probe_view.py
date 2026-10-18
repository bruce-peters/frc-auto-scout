"""Probe one uncalibrated end view: tags per frame, which tags the fixed-K consensus drops, and the
cross-view error (plus the 30 s-shifted null) of several candidate cameras. Read-only.
Usage: probe_view.py SCAN_DIR MATCH VIEW
"""
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(r"C:\Users\bruce\Documents\Coding\auto-scout")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "exp_calib" / "tools"))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from autoscout import config, field as F, pipeline as P, views  # noqa: E402
from autoscout.video import Video  # noqa: E402

src, key, vname = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
with open(src / key / "scan.pkl", "rb") as fh:
    md = pickle.load(fh)
from gate_baseline_lib import null_error  # noqa: E402

tl = md.timeline
tele = tl.teleop_start or (tl.auto_start + P.FIRST_TELEOP_GAP)
times = [float(t) for t in np.arange(tl.auto_start - 1, tele + config.TELEOP_LEN + 2, 4.0)]
vid = Video(config.VIDEOS / f"{key}.mp4")
view = next(v for v in views.PRODUCTIONS[md.prod] if v.name == vname)
fr = [(t, vid.frame_at(t)) for t in times]
fr = [(t, f) for t, f in fr if f is not None]
times = [t for t, _ in fr]
crops = [view.crop(f) for _, f in fr]
masks = view.mask_boxes(vid.width, vid.height)
tags = F.load_tags()
prior = F.load_calib_prior(md.info.event_key, vname, crops[0].shape[1::-1], exclude=key)
print("prior", {k: (np.round(v, 3).tolist() if k in ("centre", "dist") else v) for k, v in prior.items() if k not in ("K", "rvec")})
K, D = np.array(prior["K"]), np.array(prior["dist"])
dets = [F._detect_masked(c, masks) for c in crops]
print("tags per frame", Counter(len(d) for d in dets), "ids", Counter(i for d in dets for i in d))
# per-tag error under the fixed-K pan consensus about the event centre, and about the anchors' centre
diag = {}
fb = F.calibrate_view_fixed(crops, times, tags, masks, prior, diag=diag)
print("fixed_K diag", json.dumps(diag))
C = np.array(diag["centre"])
drop = Counter()
for t, d in zip(times, dets):
    if not d:
        continue
    r = F._pan_pose(d, tags, K, D, C, 3.0)
    if r is None:
        drop["frame"] += 1
        continue
    for i in d:
        if i not in r[1]:
            drop[i] += 1
print("tags dropped by the consensus (count of frames):", dict(drop))
# per-tag residual after fitting the rotation on the consensus tags
res = {}
for t, d in zip(times, dets):
    r = F._pan_pose(d, tags, K, D, C, 3.0) if d else None
    if r is None:
        continue
    tv = -cv2.Rodrigues(r[0])[0] @ C
    for i in d:
        res.setdefault(i, []).append(F._proj_err(F.tag_corners(tags[i]), d[i], r[0], tv, K, D))
print("per-tag reprojection error px (median):", {i: round(float(np.median(v)), 2) for i, v in sorted(res.items())})

cands = {"fixed_K": fb}
cands["fixed_K_eventC"] = F.calibrate_view_fixed(crops, times, tags, masks, dict(prior, centre=prior["centre"]), diag=None) \
    if diag.get("centre_from") == "event" else None
pr2 = dict(prior)
d2 = {}
F.PRIOR_MAX_SHIFT, keep_shift = 0.0, F.PRIOR_MAX_SHIFT  # force the event centre
cands["fixed_K_eventC"] = F.calibrate_view_fixed(crops, times, tags, masks, prior, diag=d2)
F.PRIOR_MAX_SHIFT = keep_shift
cands["fixed_K_pnp"] = F.calibrate_view_fixed(crops, times, tags, masks, prior, mode="pnp")
ev = F.calibrate_view_dynamic(crops, times, tags, masks, prior=prior)
cands["event_K_fit"] = ev
# free intrinsics on the consensus tags only (outlier tags removed), started from the event K
keep_ids = [i for i in res if np.median(res[i]) <= 3.0]
crops_clean = crops
plain_clean = None
try:
    obj_l, img_l, t_l = [], [], []
    for t, d in zip(times, dets):
        dd = {i: c for i, c in d.items() if i in keep_ids}
        if len(dd) < 2:
            continue
        obj_l.append(np.concatenate([F.tag_corners(tags[i]) for i in dd]).astype(np.float32))
        img_l.append(np.concatenate([dd[i] for i in dd]).astype(np.float32))
        t_l.append(t)
    w, h = crops[0].shape[1], crops[0].shape[0]
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO |
             cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    rms, K2, D2, rv, tv = cv2.calibrateCamera(obj_l, img_l, (w, h), K.copy(), D.copy(), flags=flags)
    plain_clean = F.CameraTrack(K2.tolist(), D2.ravel().tolist(), t_l, [r.ravel().tolist() for r in rv],
                                [x.ravel().tolist() for x in tv], (w, h), float(rms), len(keep_ids))
    print(f"free fit on consensus tags {keep_ids}: rms {rms:.2f} f {K2[0, 0]:.0f} k1 {D2.ravel()[0]:.3f} "
          f"centre {np.round(F.track_centre(plain_clean), 2).tolist()} frames {len(t_l)}")
except cv2.error as e:
    print("free fit failed", e)
cands["free_on_consensus"] = plain_clean
for name, cam in cands.items():
    if cam is None:
        print(f"{name:18s} none")
        continue
    g = P.cross_view_error(md, vname, cam)
    nl = null_error(md, vname, cam)
    print(f"{name:18s} rms {cam.rms:.2f} poses {len(cam.times)} f {cam.K[0][0]:.0f} centre "
          f"{np.round(F.track_centre(cam), 2).tolist()} cross-view {g['median']:.2f} / q75 {g['q75']:.2f} (n {g['n']}) "
          f"null {nl if nl is None else round(nl, 2)}", flush=True)
# the other end view and main for reference
for v2, c2 in md.cams.items():
    if v2 != "main":
        g = P.cross_view_error(md, v2, c2)
        print(f"own {v2}: cross-view {g['median']:.2f} (n {g['n']}), null {round(null_error(md, v2, c2), 2)}")
