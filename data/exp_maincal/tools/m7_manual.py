"""Reference camera for the Chezy main view: the static fit plus two hand-picked floor points (where the white
centre line meets the far and the near wall base, read off 2026cc_qm3 at t 8 s), to see what the tags alone
miss. Prints f / centre / off-field / cross-view / main-vs-end regression, and draws the field.
Usage: m7_manual.py match [t]
"""
import sys

import cv2
import numpy as np

from common import EXP, F, M, P, TAGS, fake_md, load_extract, main_boxes, off_share, xview
from m6_pairs import pairs
from autoscout import config, views
from autoscout.video import Video

from common import MANUAL_PTS as PTS

k = sys.argv[1]
t = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
ex = load_extract(k)
plain = ex["cams"]["main"]
obs = [(tt, d) for tt, d in ex["tag_obs"] if d]
bx = main_boxes(ex)
md = fake_md(ex)
K, D, C = M.track_start(plain)
cams = {"plain": plain, "static": M.fit_static(obs, TAGS, ex["size"], K, D, C),
        "manual": M.fit_static(obs, TAGS, ex["size"], K, D, C, points=PTS, w_points=2.0)}
v = Video(config.VIDEOS / f"{k}.mp4")
img = next(x for x in views.PRODUCTIONS[ex["prod"]] if x.name == "main").crop(v.frame_at(t))
v.close()
panels = []
for name, cam in cams.items():
    a = pairs(ex, cam)
    by = np.polyfit(a[:, 1], a[:, 3], 1)
    xy_pts = cam.at(t).backproject(PTS[1], z=0.0)
    print(f"{name}: f {cam.K[0][0]:.0f} k1 {cam.dist[0]:+.3f} C {np.round(F.track_centre(cam), 2).tolist()} rms {cam.rms:.2f} "
          f"off {off_share(cam, bx):.3f} xv {xview(md, cam)} | main y = {by[1]:+.2f} + {by[0]:.3f} end y | "
          f"line ends -> {np.round(xy_pts, 2).tolist()}")
    for lo, hi in ((0, 2), (2, 4), (4, 6), (6, 8.1)):
        s = (a[:, 1] >= lo) & (a[:, 1] < hi)
        if s.sum() > 20:
            print(f"    end y {lo}-{hi}: n {s.sum():5d}  main-end dy {np.median(a[s, 3] - a[s, 1]):+.2f}  dx {np.median(a[s, 2] - a[s, 0]):+.2f}")
    c = cam.at(t)
    out = F.draw_field(img, c)
    for tid in (2, 11, 21, 24):
        cv2.polylines(out, [np.int32(c.project(F.tag_corners(TAGS[tid])))], True, (0, 255, 0), 1)
    for al in ("red", "blue"):
        cv2.polylines(out, [np.int32(c.project(F.hub_box_3d(al)[:4]))], True, (255, 0, 255), 1)
    cv2.putText(out, f"{name} f {cam.K[0][0]:.0f} C {np.round(F.track_centre(cam), 1).tolist()}", (20, 620),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    panels.append(out)
cv2.imwrite(str(EXP / "img" / f"manual_{k}_{t:.0f}.jpg"), np.vstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 85])
