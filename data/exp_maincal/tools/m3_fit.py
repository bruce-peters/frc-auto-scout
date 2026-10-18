"""Static main-camera fits on the extracts: tags only vs tags + floor, per match.

Per match: the stored (plain) camera, the static fit without and with the floor term; for each: focal length,
camera centre, tag rms, off-field share of the main boxes, cross-view median vs both end cameras.
The camera is the same physical camera in every Chezy match, so the spread of f / centre over matches is
itself a check of how well each variant is determined.
Usage: m3_fit.py [prefix-or-keys ...]   (writes data/exp_maincal/m3_fit.json)
"""
import json
import sys
import time

import numpy as np

from common import EXP, F, M, P, TAGS, fake_md, keys, load_extract, main_boxes, off_share, xview

args = sys.argv[1:] or ["2026cc"]
ALL = keys()
ks = [k for a in args for k in ([a] if a in ALL else keys(a))]
rows = []
for k in ks:
    tic = time.time()
    ex = load_extract(k)
    plain = ex["cams"].get("main")
    if plain is None:
        print(k, "no main camera")
        continue
    obs = [(t, d) for t, d in ex["tag_obs"] if d]
    bx = main_boxes(ex)
    md = fake_md(ex)
    K, D, C = M.track_start(plain)
    row = {"key": k}

    def report(name, cam):
        c = F.track_centre(cam)
        row[name] = {"f": round(cam.K[0][0], 1), "k1": round(cam.dist[0], 3), "C": np.round(c, 2).tolist(),
                     "rms": round(cam.rms, 3), "off": round(off_share(cam, bx), 4), "xv": xview(md, cam)}
        if hasattr(cam, "static"):
            row[name]["start"] = cam.static.get("start")

    report("plain", plain)
    tr0 = M.fit_static(obs, TAGS, ex["size"], K, D, C)
    if tr0 is not None:
        report("static", tr0)
    cands = M.static_candidates(obs, TAGS, ex["size"], plain=plain, boxes=bx, half_depth=P.ROBOT_HALF_DEPTH)
    for name, cam in cands:
        report(name, cam)
    rows.append(row)
    s = " | ".join(f"{n}: f {v['f']:.0f} C {v['C']} rms {v['rms']:.2f} off {v['off']:.3f} xv "
                   f"{' '.join(str(x[0]) for x in v['xv'].values())}" for n, v in row.items() if n != "key")
    print(f"{k} ({time.time() - tic:.0f}s) {s}", flush=True)
json.dump(rows, open(EXP / f"m3_fit_{args[0]}.json", "w"), indent=1)
