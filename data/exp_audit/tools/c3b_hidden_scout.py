"""Cycle 3b: calibrate HIDDEN / PRIOR against the Chezy scouting.

For every scouted robot, on a 0.5 s grid while it is in its zone: state = hidden (no fused fix within
0.3 s) / visible with a detected launch within +-1 s / visible without. Scouted shot rate (balls/s,
scout taps shifted 0.8 s earlier to launch time, timer = auto_start + t) per state, and the detected
launch evidence per scouted ball while visible. From those, the HIDDEN / PRIOR values that would
give a hidden / visible-idle robot the evidence it would have shown if seen.
Usage: c3b_hidden_scout.py [SRC]  (default cc_v2)
"""
import json
import sys
from collections import defaultdict

import numpy as np

from common import A, AUDIT, DIRS, F, P, TL, load, path_video_times, resplit, scout_metrics, share_metrics, tba, team_signals
from autoscout import scouting

src = sys.argv[1] if len(sys.argv) > 1 else "cc_v2"
TAP = 0.8
bym = scouting.by_match(scouting.load())
KER_INT = (A.LAG_HI - A.LAG_LO) / 2  # integral of the triangular kernel (s)

acc = defaultdict(lambda: np.zeros(3))  # state -> (samples, scouted balls, conf-weighted launches)
per_robot = []
R, INFO, SIG = {}, {}, {}
for key, teams in sorted(bym.items()):
    if not (DIRS[src] / key / "result.json").exists():
        continue
    r = load(key, src); info = tba.match(key); tl = TL(r)
    R[key], INFO[key] = r, info
    sig = SIG[key] = team_signals(r, info)
    for al in ("red", "blue"):
        for t in info.teams(al):
            rows = teams.get(t, [])
            if not rows:
                continue
            ts = sig[t]
            if not len(ts.zone_t):
                continue
            sc = [tl.auto_start + x.shots - TAP for x in rows]
            launches = np.array([(s["t"], s["conf"]) for s in r["shots"] if s["team"] == t and s["toward"]]).reshape(-1, 2)
            st_counts = defaultdict(lambda: np.zeros(3))
            for g, z, v in zip(ts.zone_t, ts.zone_v, ts.vis_v):
                if not (z == 1.0):
                    continue
                nb = np.mean([((s >= g - 0.25) & (s < g + 0.25)).sum() for s in sc])  # balls in this 0.5 s cell
                nl = launches[(launches[:, 0] >= g - 0.25) & (launches[:, 0] < g + 0.25)]
                near = ((launches[:, 0] >= g - 1.0) & (launches[:, 0] < g + 1.0)).any()
                state = "hidden" if v < 0.5 else ("visible_launch" if near else "visible_none")
                for d in (acc[state], st_counts[state]):
                    d += (1, nb, nl[:, 1].sum())
            row = {"match": key, "team": t, "alliance": al, "scout": float(np.mean([x.n for x in rows]))}
            for s_, v in st_counts.items():
                row[s_] = v.tolist()
            per_robot.append(row)

print(f"{len(R)} matches ({src}), {len(per_robot)} scouted robots; in-zone 0.5 s samples by state:")
vis_balls = acc["visible_launch"][1] + acc["visible_none"][1]
vis_l = acc["visible_launch"][2] + acc["visible_none"][2]
recall_w = vis_l / max(vis_balls, 1e-9)
for s_, (n, b, l) in acc.items():
    print(f"   {s_:15s} {n:6.0f} samples ({n / 2:5.0f} s)  scouted {b / (n / 2):.2f} balls/s  "
          f"detected launch weight {l / (n / 2):.2f}/s  ({100 * b / sum(v[1] for v in acc.values()):.0f} % of scouted balls)")
print(f"   visible: detected launch weight per scouted ball = {recall_w:.3f}")
rate_h = acc["hidden"][1] / (acc["hidden"][0] / 2)
rate_vn = acc["visible_none"][1] / (acc["visible_none"][0] / 2)
# split_increments evidence for a robot launching at rate lam with conf c: lam*c*KER_INT. A hidden robot
# should get the evidence it would have produced if seen: recall_w * scouted rate * KER_INT.
hid_opt = recall_w * rate_h * KER_INT
prior_opt = recall_w * rate_vn * KER_INT
print(f"   implied HIDDEN (evidence a hidden in-zone robot would have shown) = {hid_opt:.2f}  (current {A.HIDDEN})")
print(f"   implied PRIOR (missed launches of a visible robot with none detected) = {prior_opt:.2f}  (current {A.PRIOR})")

# per-robot recall heterogeneity (visible states pooled)
rec = []
for x in per_robot:
    b = sum(x.get(s, [0, 0, 0])[1] for s in ("visible_launch", "visible_none"))
    l = sum(x.get(s, [0, 0, 0])[2] for s in ("visible_launch", "visible_none"))
    if b >= 20:
        rec.append((l / b, x["match"], x["team"], b))
rec.sort()
rv = np.array([x[0] for x in rec])
print(f"\nper-robot launch weight per scouted ball (visible, >= 20 scouted balls, n={len(rv)}): "
      f"q10 {np.quantile(rv, .1):.3f} q25 {np.quantile(rv, .25):.3f} median {np.median(rv):.3f} "
      f"q75 {np.quantile(rv, .75):.3f} q90 {np.quantile(rv, .9):.3f}")
print("   lowest:", [(m, t, round(v, 3), int(b)) for v, m, t, b in rec[:6]])
print("   highest:", [(m, t, round(v, 3), int(b)) for v, m, t, b in rec[-6:]])

# sweep vs scouting
H0, P0 = A.HIDDEN, A.PRIOR
sweep = []
print("\nsweep vs scouting share (and OPR):")
for h in (0.0, 0.5, 1.0, 2.0, 3.0, round(hid_opt, 2)):
    for p in (0.05, 0.12, round(prior_opt, 2)):
        A.HIDDEN, A.PRIOR = h, p
        pm = {k: resplit(R[k], INFO[k], signals=SIG[k]) for k in R}
        s = scout_metrics(pm); m = share_metrics(pm)
        sweep.append({"HIDDEN": h, "PRIOR": p, **s, **m})
        print(f"   HIDDEN {h:4.2f} PRIOR {p:4.2f}: scout corr {s['scout_corr']:.3f} mae {s['scout_mae_share']:.3f} "
              f"| OPR corr {m['corr']:.3f}")
A.HIDDEN, A.PRIOR = H0, P0
json.dump({"acc": {k: v.tolist() for k, v in acc.items()}, "recall_w": recall_w, "hid_opt": hid_opt,
           "prior_opt": prior_opt, "per_robot": per_robot, "sweep": sweep},
          open(AUDIT / f"c3b_hidden_scout_{src}.json", "w"), indent=1, default=float)
