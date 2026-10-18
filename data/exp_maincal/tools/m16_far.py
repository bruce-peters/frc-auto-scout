"""Static fit with the far-wall coverage term (robots reach the far wall): does it find the camera the hand-picked
centre-line points give, without them? Per match: f, centre, tag rms, off-field, cross-view, and where the two
hand-picked pixels land (should be (8.27, 0) and (8.27, 8.07)).
Usage: m16_far.py target q match...
"""
import sys

import numpy as np

from common import F, M, MANUAL_PTS, P, TAGS, fake_md, load_extract, main_boxes, off_share, xview

target, q = float(sys.argv[1]), float(sys.argv[2])
for k in sys.argv[3:]:
    ex = load_extract(k)
    plain = ex["cams"]["main"]
    obs = [(t, d) for t, d in ex["tag_obs"] if d]
    bx = main_boxes(ex)
    md = fake_md(ex)
    K, D, C = M.track_start(plain)
    for name, kw in (("manual", {"points": MANUAL_PTS, "w_points": 2.0}),
                     ("far", {"boxes": bx, "half_depth": P.ROBOT_HALF_DEPTH, "far_target": target, "far_q": q})):
        tr = M.fit_static(obs, TAGS, ex["size"], K, D, C, **kw)
        lm = tr.at(float(tr.times[len(tr.times) // 2])).backproject(MANUAL_PTS[1], z=0.0)
        print(f"{k:16s} {name:6s} f {tr.K[0][0]:6.0f} k1 {tr.dist[0]:+.3f} C {np.round(F.track_centre(tr), 2).tolist()} "
              f"rms {tr.rms:.2f} off {off_share(tr, bx):.3f} xv {[v[0] for v in xview(md, tr).values()]} "
              f"landmarks -> {np.round(lm, 2).tolist()}", flush=True)
