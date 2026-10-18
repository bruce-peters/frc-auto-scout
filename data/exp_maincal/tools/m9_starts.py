"""Static floor fit from each start (plain / ippe): cost, f, centre, off-field. Usage: m9_starts.py match..."""
import sys
import numpy as np
from common import F, M, P, TAGS, load_extract, main_boxes, off_share
for k in sys.argv[1:]:
    ex = load_extract(k)
    plain = ex["cams"]["main"]
    obs = [(tt, d) for tt, d in ex["tag_obs"] if d]
    bx = main_boxes(ex)
    K, D, C = M.track_start(plain)
    ci = M.ippe_start(obs, TAGS, K, D)
    print(k, "ippe start centre", None if ci is None else np.round(ci, 2).tolist(), "plain centre", np.round(C, 2).tolist())
    for name, C0 in (("plain", C), ("ippe", ci)):
        if C0 is None:
            continue
        for fl in (False, True):
            tr = M.fit_static(obs, TAGS, ex["size"], K, D, C0, boxes=bx if fl else None, half_depth=P.ROBOT_HALF_DEPTH)
            n, o = M.window_off(tr, bx, P.ROBOT_HALF_DEPTH)
            bad = [(round(tr.times[i], 1), int(n[i]), int(o[i])) for i in range(len(n)) if n[i] and o[i] / n[i] > 0.2]
            print(f"   {name:5s} floor={fl}: cost {tr.static['cost']:.1f} f {tr.K[0][0]:.0f} k1 {tr.dist[0]:+.3f} C {tr.static['centre']} "
                  f"rms {tr.rms:.3f} off {off_share(tr, bx):.3f} bad windows {bad}")
