"""Field lines (as dots, only inside the view) through the stored end cameras and a main camera of a match:
are the end cameras right at long range? Usage: m14_end_overlay.py match t"""
import sys
import cv2
import numpy as np
from common import EXP, F, TAGS, load_extract
from autoscout import config, views
from autoscout.video import Video


def dots(img, cam, color):
    out = img
    h, w = img.shape[:2]
    lines = F.field_lines()
    for al in ("red", "blue"):
        b = F.hub_box_3d(al)[:4]
        lines.append(np.vstack([b, b[:1]]))
    for ln in lines:
        for a, b in zip(ln[:-1], ln[1:]):
            P = np.array([a + (b - a) * s for s in np.linspace(0, 1, 200)])
            pc = (cam.R @ (P - cam.center_world()).T).T
            P = P[pc[:, 2] > 0.5]
            if not len(P):
                continue
            p = cam.project(P)
            for u, v in p:
                if 0 <= u < w and 0 <= v < h:
                    cv2.circle(out, (int(u), int(v)), 2, color, -1)
    return out


k, t = sys.argv[1], float(sys.argv[2])
ex = load_extract(k)
v = Video(config.VIDEOS / f"{k}.mp4")
fr = v.frame_at(t)
v.close()
panels = []
for vw in views.PRODUCTIONS[ex["prod"]]:
    if vw.name == "main" or vw.name not in ex["cams"]:
        continue
    cam = ex["cams"][vw.name].at(t)
    out = dots(vw.crop(fr).copy(), cam, (0, 255, 255))
    cv2.putText(out, f"{vw.name} C {np.round(cam.center_world(), 2).tolist()} f {cam.K[0][0]:.0f} k1 {cam.dist[0]:.2f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    panels.append(out)
cv2.imwrite(str(EXP / "img" / f"end_{k}_{t:.0f}.jpg"), np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 90])
