"""Error landscape of the fixed-K pan/tilt model over the camera centre (one end view), and what the
joint centre + rotations refine does from the anchors' centre and from the event centre. Read-only.
Usage: probe_centre.py SCAN_DIR MATCH VIEW
"""
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(r"C:\Users\bruce\Documents\Coding\auto-scout")
sys.path.insert(0, str(ROOT))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from autoscout import config, field as F, pipeline as P, views  # noqa: E402
from autoscout.video import Video  # noqa: E402

src, key, vname = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
with open(src / key / "scan.pkl", "rb") as fh:
    md = pickle.load(fh)
tl = md.timeline
tele = tl.teleop_start or (tl.auto_start + P.FIRST_TELEOP_GAP)
times = [float(t) for t in np.arange(tl.auto_start - 1, tele + config.TELEOP_LEN + 2, 4.0)]
vid = Video(config.VIDEOS / f"{key}.mp4")
view = next(v for v in views.PRODUCTIONS[md.prod] if v.name == vname)
masks = view.mask_boxes(vid.width, vid.height)
dets = []
for t in times:
    f = vid.frame_at(t)
    if f is not None:
        d = F._detect_masked(view.crop(f), masks)
        if d:
            dets.append((t, d))
tags = F.load_tags()
prior = F.load_calib_prior(md.info.event_key, vname, view.crop(f).shape[1::-1], exclude=key)
K, D = np.array(prior["K"]), np.array(prior["dist"])
C0 = np.array(prior["centre"])


def total(C, fixed_ids=None):
    """rms over all frames of the pan consensus about C, and how many tag observations it keeps."""
    se, n, kept = 0.0, 0, 0
    for t, d in dets:
        dd = d if fixed_ids is None else {i: c for i, c in d.items() if i in fixed_ids}
        if not dd:
            continue
        r = F._pan_pose(dd, tags, K, D, C, 3.0)
        if r is None:
            continue
        se += r[2] ** 2 * len(r[1])
        n += len(r[1])
        kept += len(r[1])
    return np.sqrt(se / max(n, 1)), kept


print("event centre", np.round(C0, 3), "rms %.3f kept %d" % total(C0))
good = {18, 25, 26, 27, 28}
print("event centre, tags", sorted(good), "rms %.3f kept %d" % total(C0, good))
for dz in (-0.1, 0.0, 0.1, 0.2, 0.3, 0.35, 0.4):
    for dx in (-0.2, 0.0, 0.2):
        C = C0 + np.array([dx, 0, dz])
        print(f"  C {np.round(C, 2)}: rms %.3f kept %d" % total(C, good))
# anchors
pnp = [F._pnp_ransac(d, tags, K, D, 3.0) for t, d in dets if len(d) >= 2]
anch = np.array([-cv2.Rodrigues(r[0])[0].T @ r[1] for r in pnp if r is not None and len(r[2]) >= 3])
print("anchors", len(anch), "median", np.round(np.median(anch, 0), 3), "spread", np.round(anch.std(0), 3),
      "pnp rms median", np.round(np.median([r[3] for r in pnp if r is not None]), 2))
for name, Cs in (("anchors", np.median(anch, 0)), ("event", C0)):
    poses = {t: F._pan_pose({i: c for i, c in d.items() if i in good}, tags, K, D, Cs, 3.0) for t, d in dets}
    poses = {t: p for t, p in poses.items() if p is not None}
    ts = sorted(poses)
    fo = [(np.concatenate([F.tag_corners(tags[i]) for i in poses[t][1]]),
           np.concatenate([dict(dets)[t][i] for i in poses[t][1]])) for t in ts]
    from scipy.optimize import least_squares  # noqa: E402
    Cb, rvs = F._pan_bundle(fo, K, D, Cs, [poses[t][0] for t in ts])
    print(f"bundle from {name} centre {np.round(Cs, 3)} -> {np.round(Cb, 3)}: rms %.3f kept %d" % total(Cb, good))
