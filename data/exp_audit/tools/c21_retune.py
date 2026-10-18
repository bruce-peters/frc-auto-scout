"""Cycle 21: re-tune HIDDEN / PRIOR on the larger scouting set (now ~177 scouted cc team-matches vs 51 when
they were last checked), with the compression slope (video share ~ scouted share) as an extra readout.
JSON re-split of current output/ (mirrors split_increments incl. AUTO_ZONE_GATE)."""
import numpy as np

from common import A, load, match_keys, resplit, scout_counts, scout_metrics, share_metrics, tba, team_signals

R, INFO, SIG = {}, {}, {}
for k in match_keys("v2"):
    R[k] = load(k, "v2"); INFO[k] = tba.match(k); SIG[k] = team_signals(R[k], INFO[k])
sc = scout_counts()


def slope(pm):
    vs, ss = [], []
    for k, f in pm.items():
        if k not in sc:
            continue
        for al in ("red", "blue"):
            tm = INFO[k].teams(al)
            if not all(t in sc[k] for t in tm):
                continue
            v = np.array([f[t] for t in tm]); s = np.array([sc[k][t] for t in tm])
            if v.sum() > 0 and s.sum() > 0:
                vs += (v / v.sum()).tolist(); ss += (s / s.sum()).tolist()
    return float(np.polyfit(ss, vs, 1)[0])


H0, P0 = A.HIDDEN, A.PRIOR
print(f"{len(R)} matches; current HIDDEN {H0} PRIOR {P0}")
for h in (0.5, 1.0, 2.0, 3.0, 5.0):
    for p in (0.03, 0.12, 0.3):
        A.HIDDEN, A.PRIOR = h, p
        pm = {k: resplit(R[k], INFO[k], signals=SIG[k]) for k in R}
        m, s = share_metrics(pm), scout_metrics(pm)
        print(f"   HIDDEN {h:3.1f} PRIOR {p:4.2f}: scout corr {s['scout_corr']:.3f} mae {s['scout_mae_share']:.4f} slope {slope(pm):.2f} "
              f"(n={s['n']}) | OPR corr {m['corr']:.3f} top {m['top']}/{m['alliances']} MAE {m['mae']:.1f}", flush=True)
A.HIDDEN, A.PRIOR = H0, P0
