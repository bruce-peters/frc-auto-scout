"""cc_qm24, 70-74 s (video time): the blue volley where 254 stands hidden behind 3847 in the blue end camera.
Credited launches there (output/ result.json), today's robot vs the two-view pick (all six robots' fused
tracks, triangulate.pick_robot), per view and rule."""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import eval_mv as EM  # noqa: E402
from autoscout import config, triangulate as T  # noqa: E402

k, t0, t1 = "2026cc_qm24", 70.0, 74.0
if len(sys.argv) > 1:
    k, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
P = C.load_pairs(k)
res = json.loads((config.OUTPUT / k / "result.json").read_text())
trk = C.team_tracks(res)
ix, bvt = C.anchor_index(P), C.by_view_track(P)
shots = [s for s in res["shots"] if t0 <= s["t"] <= t1]
print(k, t0, t1, len(shots), "credited launches;", "today:", dict(Counter((s["view"], s["team"]) for s in shots)))
for rn in ("any fit", "rms<=10", "norm<=2 (main 8, end 4 px)", "rms<=6", "any, mutual strict"):
    for gn, gate in EM.GATES.items():
        cnt, pair_cnt = Counter(), Counter()
        for s in shots:
            a = ix.get(C.anchor_key(s["view"], [(s["t"], s["u"], s["v"])]))
            c = C.best_candidate(a, EM.RULES[rn], bvt) if a else None
            if c is None or c["launch"] is None:
                cnt[(s["view"], "no pair: " + str(s["team"]))] += 1
                continue
            L = c["launch"]
            rob = {t: C.position(arr, L["t"]) for t, arr in trk.items()}
            rob = {t: p[:2] for t, p in rob.items() if p is not None}
            r = T.pick_robot(L, rob, **gate)
            cnt[(s["view"], r["team"])] += 1
        print(f"  {rn:28s} {gn:8s}", dict(sorted(cnt.items(), key=lambda x: (x[0][0], str(x[0][1])))))
