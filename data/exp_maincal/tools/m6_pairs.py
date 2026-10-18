"""Main vs end-camera robot positions for matched robots (same alliance, same instant, nearest within 2 m):
is the main camera's floor map compressed or shifted? Regression of the main view's x and y on the end
view's, and the residual by field region.
Usage: m6_pairs.py match...
"""
import sys

import numpy as np

from common import F, M, P, TAGS, VN, load_extract, main_boxes
from autoscout import robots, views


def place(cam, d, vl, limit):
    ox, oy = vl[VN[int(d[1])]].offset(1920, 1080)
    c = cam.at(float(d[0]))
    xy = M.floor_xy(c, np.array([[(d[2] + d[4]) / 2 - ox, d[5] - oy]]), P.ROBOT_HALF_DEPTH)[0]
    if not np.all(np.isfinite(xy)):
        return None
    rng = np.hypot(*(xy - c.center_world()[:2]))
    return xy if robots.in_field(xy) and (limit is None or rng <= limit) else None


def pairs(ex, cam_main, max_d=2.0):
    vl = {v.name: v for v in views.PRODUCTIONS[ex["prod"]]}
    by_t = {}
    for d in ex["dets"]:
        by_t.setdefault(round(d[0], 3), []).append(d)
    out = []
    for ds in by_t.values():
        mains = [(d[6], p, d) for d in ds if d[1] == 0 and (p := place(cam_main, d, vl, None)) is not None]
        for d in ds:
            if d[1] == 0 or VN[int(d[1])] not in ex["cams"]:
                continue
            p = place(ex["cams"][VN[int(d[1])]], d, vl, P.END_VIEW_RANGE_COV)
            if p is None:
                continue
            q = [(np.hypot(*(p - pm)), pm, dm) for al, pm, dm in mains if al == d[6]]
            if q:
                dist, pm, dm = min(q, key=lambda x: x[0])
                if dist < max_d:
                    out.append((*p, *pm, d[1], dm[0], (dm[2] + dm[4]) / 2, dm[5]))
    return np.array(out)


if __name__ == "__main__":
    for k in sys.argv[1:]:
        ex = load_extract(k)
        plain = ex["cams"]["main"]
        obs = [(t, d) for t, d in ex["tag_obs"] if d]
        K, D, C = M.track_start(plain)
        for name, cam in (("plain", plain), ("static", M.fit_static(obs, TAGS, ex["size"], K, D, C))):
            a = pairs(ex, cam)
            if not len(a):
                continue
            bx, by = np.polyfit(a[:, 0], a[:, 2], 1), np.polyfit(a[:, 1], a[:, 3], 1)
            d = np.hypot(a[:, 0] - a[:, 2], a[:, 1] - a[:, 3])
            print(f"{k} {name}: {len(a)} pairs, median dist {np.median(d):.2f}; main x = {bx[1]:+.2f} + {bx[0]:.3f} end x; "
                  f"main y = {by[1]:+.2f} + {by[0]:.3f} end y; mean dy (main-end) {np.mean(a[:, 3] - a[:, 1]):+.2f}")
            for lo, hi in ((0, 2), (2, 4), (4, 6), (6, 8.1)):
                s = (a[:, 1] >= lo) & (a[:, 1] < hi)
                if s.sum() > 20:
                    print(f"    end y {lo}-{hi}: n {s.sum():5d}  main-end dy {np.median(a[s, 3] - a[s, 1]):+.2f}  dx {np.median(a[s, 2] - a[s, 0]):+.2f}")
