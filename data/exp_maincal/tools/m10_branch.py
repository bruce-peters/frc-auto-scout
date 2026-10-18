"""Are the stored main-camera rvecs on different branches (|r| near pi, consecutive rvecs far apart although the
rotation between them is small)? Linear interpolation (CameraTrack.at) between such rvecs gives garbage poses.
Usage: m10_branch.py [prefix]"""
import sys
import cv2
import numpy as np
from common import F, M, P, keys, load_extract, main_boxes, off_share
for k in keys(sys.argv[1] if len(sys.argv) > 1 else ""):
    ex = load_extract(k)
    cam = ex["cams"].get("main")
    if cam is None or not hasattr(cam, "rvecs"):
        continue
    r = np.array(cam.rvecs)
    nr = np.linalg.norm(r, axis=1)
    flips = 0
    for a, b in zip(r[:-1], r[1:]):
        ang = F._rot_angle(a, b)
        if np.linalg.norm(a - b) > 0.5 and ang < 10:
            flips += 1
    bx = main_boxes(ex)
    # same poses, interpolation made branch-consistent (each rvec moved to the branch nearest the previous one)
    r2 = [r[0]]
    for x in r[1:]:
        th = np.linalg.norm(x)
        alt = x * (1 - 2 * np.pi / th) if th > 1e-9 else x
        r2.append(alt if np.linalg.norm(alt - r2[-1]) < np.linalg.norm(x - r2[-1]) else x)
    # tvec must be consistent with the same R: the rotation is the same for both branches, so tvecs stay
    fixed = F.CameraTrack(cam.K, cam.dist, cam.times, [x.tolist() for x in r2], cam.tvecs, cam.size, cam.rms, cam.n_tags)
    print(f"{k:18s} |r| {nr.min():.3f}-{nr.max():.3f}  branch flips {flips:2d}/{len(r) - 1}  off {off_share(cam, bx):.3f} -> "
          f"branch-consistent {off_share(fixed, bx):.3f}", flush=True)
