"""Auto credit that moves when zone gating is turned off: who gains it (zone fraction, position, visibility)."""
import json
from collections import defaultdict
import numpy as np
from common import A, AUDIT, F, P, TL, load, match_keys, path_video_times, tba, team_signals
ZL = {"blue": F.ALLIANCE_ZONE_DEPTH, "red": F.L - F.ALLIANCE_ZONE_DEPTH}
cat = defaultdict(float); depth_gain = []; t_gain = []
tot_auto = 0.0
for k in match_keys("v2"):
    r = load(k, "v2"); info = tba.match(k); tl = TL(r); sig = team_signals(r, info)
    for al in ("red", "blue"):
        if not tl.counts.get(al):
            continue
        tm = info.teams(al)
        paths = {t: path_video_times(r["paths"].get(str(t), []), tl) for t in tm}
        for t, n in A.increments_from_counts(tl.counts[al]):
            mt = tl.match_time(t - 2.0)
            if mt is None or A.period_of(mt) != "auto":
                continue
            tot_auto += n
            w_on, w_off, zz = [], [], []
            for team in tm:
                ts = sig[team]
                z = ts.zone_frac(t - A.LAG_HI, t - A.LAG_LO)
                s = float((A.kernel(t - ts.shots) * ts.shot_w).sum()) if len(ts.shots) else 0.0
                vis = ts.vis_frac(t - A.LAG_HI, t - A.LAG_LO)
                ph = A.PRIOR * vis + A.HIDDEN * (1 - vis)
                w_on.append(z * (s + ph)); w_off.append(s + ph); zz.append((z, s > 0.05, vis))
            w_on, w_off = np.array(w_on), np.array(w_off)
            w_on = w_on / w_on.sum() if w_on.sum() > 1e-9 else np.ones(3) / 3
            w_off = w_off / w_off.sum()
            for i, team in enumerate(tm):
                g = n * (w_off[i] - w_on[i])
                if g <= 0:
                    continue
                z, has_l, vis = zz[i]
                key = ("z<0.2" if z < 0.2 else "z 0.2-0.8" if z < 0.8 else "z>=0.8") + (" +launches" if has_l else " no launches") + \
                      (" hidden" if vis < 0.5 else " visible")
                cat[key] += g
                p = P._track_position(paths[team], t - 2.8) if len(paths[team]) else None
                if p is not None and z < 0.2:
                    depth_gain.append(((ZL["blue"] - p[0]) if al == "blue" else (p[0] - ZL["red"]), g))
                if z < 0.2:
                    t_gain.append((mt, g))
moved = sum(cat.values())
print(f"auto balls {tot_auto:.0f}; credit moved by ungating: {moved:.0f} ({100 * moved / tot_auto:.1f} % of auto balls)")
for c, v in sorted(cat.items(), key=lambda x: -x[1]):
    print(f"   gained by robots with {c:32s} {v:6.0f} ({100 * v / moved:4.1f} %)")
d = np.array(depth_gain)
if len(d):
    w = d[:, 1]
    print("   z<0.2 gainers: position 2.8 s before the count, depth into own zone (m), credit-weighted quantiles 10/50/90:",
          np.round([d[np.argsort(d[:, 0])][np.searchsorted(np.cumsum(w[np.argsort(d[:, 0])]) / w.sum(), q), 0] for q in (.1, .5, .9)], 2))
tg = np.array(t_gain)
if len(tg):
    for a, b in ((0, 5), (5, 10), (10, 15), (15, 23)):
        m = (tg[:, 0] >= a) & (tg[:, 0] < b)
        print(f"   z<0.2 credit gained at match t {a:2d}-{b:2d} s: {tg[m, 1].sum():.0f}")
