"""Quantiles of the main-view robot floor points under the static (tags only) and the manual-point camera:
do robots reach the walls (a two-sided 'coverage' constraint)? Usage: m8_quant.py match..."""
import sys
import numpy as np
from common import F, M, P, TAGS, load_extract, main_boxes
from common import MANUAL_PTS as PTS
for k in sys.argv[1:]:
    ex = load_extract(k)
    plain = ex["cams"]["main"]
    obs = [(tt, d) for tt, d in ex["tag_obs"] if d]
    bx = main_boxes(ex)
    K, D, C = M.track_start(plain)
    for name, cam in (("static", M.fit_static(obs, TAGS, ex["size"], K, D, C)),
                      ("manual", M.fit_static(obs, TAGS, ex["size"], K, D, C, points=PTS, w_points=2.0))):
        xy = np.concatenate([M.floor_xy(cam.at(float(t)), bx[bx[:, 0] == t, 1:], P.ROBOT_HALF_DEPTH) for t in np.unique(bx[:, 0])])
        q = [0.001, 0.005, 0.01, 0.05, 0.5, 0.95, 0.99, 0.995, 0.999]
        print(f"{k} {name}: f {cam.K[0][0]:.0f} x q {np.round(np.nanquantile(xy[:, 0], q), 2).tolist()}\n      y q {np.round(np.nanquantile(xy[:, 1], q), 2).tolist()}")
