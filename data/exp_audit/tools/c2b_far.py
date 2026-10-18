"""Credited launches whose shooter track is far outside its zone: by link type / view / x, and whether
the alliance counter rises ~0.8-5.5 s later (feeds/passes from the neutral zone should not)."""
import json, sys
from collections import Counter
import numpy as np
from common import A, AUDIT, F, P, TL, load, match_keys, path_video_times, tba
src = sys.argv[1] if len(sys.argv) > 1 else "v2"
ZL = {"blue": F.ALLIANCE_ZONE_DEPTH, "red": F.L - F.ALLIANCE_ZONE_DEPTH}
cnt = Counter(); xs = []; rows = []
resp = {"in": [], "far": []}
for k in match_keys(src):
    r = load(k, src); info = tba.match(k); tl = TL(r)
    inc = {al: A.increments_from_counts(tl.counts[al]) for al in ("red", "blue") if tl.counts.get(al)}
    for s in r["shots"]:
        if s["team"] is None or not s["toward"]:
            continue
        al = "red" if s["team"] in info.red else "blue"
        pts = path_video_times(r["paths"].get(str(s["team"]), []), tl)
        p = P._track_position(pts, s["t"]) if len(pts) else None
        if p is None:
            continue
        d = (ZL["blue"] - p[0]) if al == "blue" else (p[0] - ZL["red"])
        # counter balls 0.8-5.5 s later (kernel weighted)
        c = sum(n * A.kernel(t - s["t"]) for t, n in inc.get(al, [])) if al in inc else np.nan
        if d < -1.5:
            cnt[(s["how"], s["view"], "obs" if p[2] else "interp")] += 1
            xs.append(p[0] if al == "blue" else F.L - p[0])  # distance from own wall
            resp["far"].append(c)
            rows.append({"key": k, "t": s["t"], "mt": tl.match_time(s["t"]), "team": s["team"], "al": al,
                         "x": round(p[0], 2), "y": round(p[1], 2), "how": s["how"], "view": s["view"], "u": s["u"], "v": s["v"]})
        elif d >= 0:
            resp["in"].append(c)
n = sum(cnt.values())
print(f"far-out (> 1.5 m outside) credited launches: {n}")
for kk, v in cnt.most_common(12):
    print("  ", kk, v)
xs = np.array(xs)
print("distance from own alliance wall (m) of far-out shooters: quantiles 10/25/50/75/90:",
      np.round(np.quantile(xs, [.1, .25, .5, .75, .9]), 1), " (zone line at 4.03, centre 8.27, far zone line 12.5)")
print(f"kernel-weighted counter balls after a launch: in-zone mean {np.nanmean(resp['in']):.2f}, far-out mean {np.nanmean(resp['far']):.2f}")
json.dump(rows, open(AUDIT / f"c2b_far_{src}.json", "w"), indent=0, default=float)
