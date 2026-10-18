"""Where do the main boxes land under each candidate camera (one match)? x/y quantiles, off-field boxes by edge,
off-field boxes by image row, and the floor-term cost pieces.
Usage: m4_where.py match
"""
import sys

import numpy as np

from common import F, M, P, TAGS, load_extract, main_boxes

k = sys.argv[1]
ex = load_extract(k)
plain = ex["cams"]["main"]
obs = [(t, d) for t, d in ex["tag_obs"] if d]
bx = main_boxes(ex)
K, D, C = M.track_start(plain)
cams = {"plain": plain, "static": M.fit_static(obs, TAGS, ex["size"], K, D, C),
        "static_floor": M.fit_static(obs, TAGS, ex["size"], K, D, C, boxes=bx, half_depth=P.ROBOT_HALF_DEPTH)}
for name, cam in cams.items():
    xy = np.array([M.floor_xy(cam.at(float(t)), np.array([[u, v]]), P.ROBOT_HALF_DEPTH)[0] for t, u, v in bx])
    off = M.off_field(cam, bx, P.ROBOT_HALF_DEPTH)
    fin = np.all(np.isfinite(xy), axis=1)
    q = lambda a: np.round(np.quantile(a, [0.005, 0.05, 0.5, 0.95, 0.995]), 2).tolist()
    print(f"== {name}: f {cam.K[0][0]:.0f} k1 {cam.dist[0]:.3f} C {np.round(F.track_centre(cam), 2).tolist()} "
          f"rms {cam.rms:.2f} off {off.mean():.3f} nonfinite {np.mean(~fin):.3f}")
    print(f"   x q {q(xy[fin, 0])}  y q {q(xy[fin, 1])}")
    o = xy[off & fin]
    if len(o):
        L, W = M.L, M.W
        print("   off beyond: x<0 %.2f x>L %.2f y<0 %.2f y>W %.2f" % (np.mean(o[:, 0] < -0.4), np.mean(o[:, 0] > L + 0.4),
                                                               np.mean(o[:, 1] < -0.4), np.mean(o[:, 1] > W + 0.4)))
        print("   off boxes image v q", np.round(np.quantile(bx[off, 2], [0.1, 0.5, 0.9]), 0).tolist(), " u q",
              np.round(np.quantile(bx[off, 1], [0.1, 0.5, 0.9]), 0).tolist())
    # where do the four field corners and the centre line ends project?
    pts = np.array([[0, 0, 0], [M.L, 0, 0], [M.L, M.W, 0], [0, M.W, 0], [M.L / 2, 0, 0], [M.L / 2, M.W, 0]], float)
    print("   field corners (x0y0, xLy0, xLyW, x0yW, mid y0, mid yW) ->", np.round(cam.project(pts), 0).astype(int).tolist())
