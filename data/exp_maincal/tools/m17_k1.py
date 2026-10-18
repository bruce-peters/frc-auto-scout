"""Tags-only static fit with k1 held at 0 (and with the far-wall term): does fixing the distortion pin the
focal length / elevation that the hand-picked landmarks give? Usage: m17_k1.py match..."""
import sys
import numpy as np
from common import F, M, MANUAL_PTS, P, TAGS, fake_md, load_extract, main_boxes, off_share, xview
for k in sys.argv[1:]:
    ex = load_extract(k)
    plain = ex["cams"]["main"]
    obs = [(t, d) for t, d in ex["tag_obs"] if d]
    bx = main_boxes(ex)
    md = fake_md(ex)
    K, D, C = M.track_start(plain)
    D0 = np.zeros(5)
    for name, kw in (("static", {}), ("k1=0", {"fix_k1": True}),
                     ("k1=0 far", {"fix_k1": True, "boxes": bx, "half_depth": P.ROBOT_HALF_DEPTH, "far_target": 0.55}),
                     ("manual k1=0", {"fix_k1": True, "points": MANUAL_PTS, "w_points": 2.0})):
        tr = M.fit_static(obs, TAGS, ex["size"], K, D if name == "static" else D0, C, **kw)
        lm = tr.at(float(tr.times[len(tr.times) // 2])).backproject(MANUAL_PTS[1], z=0.0)
        print(f"{k:16s} {name:11s} f {tr.K[0][0]:6.0f} k1 {tr.dist[0]:+.3f} C {np.round(F.track_centre(tr), 2).tolist()} "
              f"rms {tr.rms:.2f} off {off_share(tr, bx):.3f} xv {[v[0] for v in xview(md, tr).values()]} "
              f"landmarks -> {np.round(lm, 2).tolist()}", flush=True)
