"""Draw the field (perimeter, centre line, zone lines, hub outlines + openings) through candidate main cameras
on a main-view frame, to see which camera matches the visible field.
Usage: m5_overlay.py match t   (writes data/exp_maincal/img/<match>_<t>.jpg, one panel per camera)
"""
import sys

import cv2
import numpy as np

from common import EXP, F, M, P, TAGS, load_extract, main_boxes
from autoscout import config, views
from autoscout.video import Video

k, t = sys.argv[1], float(sys.argv[2])
ex = load_extract(k)
plain = ex["cams"]["main"]
obs = [(tt, d) for tt, d in ex["tag_obs"] if d]
bx = main_boxes(ex)
K, D, C = M.track_start(plain)
cams = {"plain": plain, "static": M.fit_static(obs, TAGS, ex["size"], K, D, C),
        "static_floor": M.fit_static(obs, TAGS, ex["size"], K, D, C, boxes=bx, half_depth=P.ROBOT_HALF_DEPTH)}
for extra in sys.argv[3:]:  # name=f,cx,cy,cz : static fit with this focal length / centre start, intrinsics free
    name, vals = extra.split("=")
    f, cx_, cy_, cz_ = map(float, vals.split(","))
    K2 = K.copy(); K2[0, 0] = K2[1, 1] = f
    cams[name] = M.fit_static(obs, TAGS, ex["size"], K2, D * 0, np.array([cx_, cy_, cz_]), fix_intrinsics=True)
v = Video(config.VIDEOS / f"{k}.mp4")
img = next(x for x in views.PRODUCTIONS[ex["prod"]] if x.name == "main").crop(v.frame_at(t))
v.close()
panels = []
for name, cam in cams.items():
    c = cam.at(t)
    out = F.draw_field(img, c)
    for tid in (2, 11, 21, 24, 5, 8, 18, 27):
        p = c.project(F.tag_corners(TAGS[tid]))
        cv2.polylines(out, [np.int32(p)], True, (0, 255, 0), 1)
    cv2.putText(out, f"{name} f {cam.K[0][0]:.0f} C {np.round(F.track_centre(cam), 1).tolist()}", (20, 620),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    panels.append(out)
(EXP / "img").mkdir(exist_ok=True)
cv2.imwrite(str(EXP / "img" / f"{k}_{t:.0f}.jpg"), np.vstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 85])
print("wrote", EXP / "img" / f"{k}_{t:.0f}.jpg")
