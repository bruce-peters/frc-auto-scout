"""Do the end cameras let robots reach the far wall (y = 0)? Quantiles of end-view robot y (placed as analyse:
range <= END_VIEW_RANGE_COV), vs the main camera (static / manual landmarks). Usage: m18_endq.py match..."""
import sys
import numpy as np
from common import F, M, MANUAL_PTS, P, TAGS, VN, load_extract, main_boxes
from autoscout import robots, views
q = [0.002, 0.005, 0.01, 0.02, 0.05]
for k in sys.argv[1:]:
    ex = load_extract(k)
    vl = {v.name: v for v in views.PRODUCTIONS[ex["prod"]]}
    for vi in (1, 2):
        vn = VN[vi]
        if vn not in ex["cams"]:
            continue
        d = ex["dets"][ex["dets"][:, 1] == vi]
        ox, oy = vl[vn].offset(1920, 1080)
        ys = []
        for r in d:
            c = ex["cams"][vn].at(float(r[0]))
            xy = M.floor_xy(c, np.array([[(r[2] + r[4]) / 2 - ox, r[5] - oy]]), P.ROBOT_HALF_DEPTH)[0]
            if np.all(np.isfinite(xy)) and robots.in_field(xy) and np.hypot(*(xy - c.center_world()[:2])) <= P.END_VIEW_RANGE_COV:
                ys.append(xy[1])
        print(f"{k:16s} {vn:9s} n {len(ys):5d} y q {np.round(np.quantile(ys, q), 2).tolist()}")
    plain = ex["cams"]["main"]
    obs = [(t, dd) for t, dd in ex["tag_obs"] if dd]
    bx = main_boxes(ex)
    K, D, C = M.track_start(plain)
    for name, kw in (("static", {}), ("manual", {"points": MANUAL_PTS, "w_points": 2.0})):
        tr = M.fit_static(obs, TAGS, ex["size"], K, D, C, **kw)
        xy = np.concatenate([M.floor_xy(tr.at(float(t)), bx[bx[:, 0] == t, 1:], P.ROBOT_HALF_DEPTH) for t in np.unique(bx[:, 0])])
        ok = np.all(np.isfinite(xy), axis=1) & np.array([robots.in_field(p) if np.all(np.isfinite(p)) else False for p in xy])
        print(f"{k:16s} main {name:6s} n {ok.sum():5d} y q {np.round(np.quantile(xy[ok, 1], q), 2).tolist()}")
