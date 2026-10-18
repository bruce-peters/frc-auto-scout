"""Where do Chezy main-view floor points land when they are off the field? (qm5, qm26, qm15 for contrast)"""
import gc, pickle, sys
import numpy as np
from common import DIRS, P, config
from autoscout import robots, views
for k in sys.argv[1:]:
    with open(DIRS["v2"] / k / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    v = next(x for x in views.PRODUCTIONS[md.prod] if x.name == "main"); ox, oy = v.offset(1920, 1080)
    cam = md.cams["main"]
    xs = []
    for d in md.dets:
        if d.view != "main":
            continue
        c = cam.at(d.t)
        u, vv = (d.box[0] + d.box[2]) / 2 - ox, d.box[3] - oy
        xy = c.backproject(np.array([[u, vv]]), z=0.03)[0]
        if np.all(np.isfinite(xy)):
            xs.append((xy[0], xy[1], u, vv, robots.in_field(xy)))
    a = np.array(xs, float)
    off = a[a[:, 4] == 0]
    on = a[a[:, 4] == 1]
    print(f"{k}: {len(a)} main boxes, off {len(off)}")
    for name, s in (("in-field", on), ("off-field", off)):
        if len(s):
            print(f"   {name:9s} field x q10/50/90 {np.round(np.quantile(s[:, 0], [.1, .5, .9]), 1)}  y q10/50/90 "
                  f"{np.round(np.quantile(s[:, 1], [.1, .5, .9]), 1)}  image v (row) median {np.median(s[:, 3]):.0f}  u median {np.median(s[:, 2]):.0f}")
    # off-field: beyond which edge?
    L, W = config.FIELD_LENGTH, config.FIELD_WIDTH
    if len(off):
        e = {"y > W": np.mean(off[:, 1] > W + 0.4), "y < 0": np.mean(off[:, 1] < -0.4), "x < 0": np.mean(off[:, 0] < -0.4), "x > L": np.mean(off[:, 0] > L + 0.4)}
        print("   off-field beyond:", {kk: round(float(vv), 2) for kk, vv in e.items()})
    del md; gc.collect()
