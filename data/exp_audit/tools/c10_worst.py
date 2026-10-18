"""Cycle 10: worst per-team errors vs the Chezy scouting on the current output/ results, with diagnostics.

Per scouted team-match (all three robots of the alliance scouted): video share - scout share. For the
worst ones: credited launches (box/proj), credit sources (launch / prior / hidden), visible fraction, share
of its track time on segments whose own reads name another team or that have no reads, and how far apart
its two scouts are.
"""
import json
from collections import defaultdict

import numpy as np

from common import A, AUDIT, TL, load, match_keys, scout_counts, tba, team_signals
from autoscout import scouting

bym = scouting.by_match(scouting.load())
sc = scout_counts()
rows = []
for k in match_keys("v2"):
    if k not in sc:
        continue
    r = load(k, "v2"); info = tba.match(k); tl = TL(r)
    sig = team_signals(r, info)
    for al in ("red", "blue"):
        tm = info.teams(al)
        if not all(t in sc[k] for t in tm):
            continue
        v = np.array([r["teams"][str(t)]["fuel"] for t in tm]); s = np.array([sc[k][t] for t in tm])
        if v.sum() <= 0 or s.sum() <= 0:
            continue
        # credit sources per team
        src = np.zeros((3, 3))
        if tl.counts.get(al):
            for t, n in A.increments_from_counts(tl.counts[al]):
                mt = tl.match_time(t - 2.0)
                per = A.period_of(mt) if mt is not None else "x"
                p = []
                for team in tm:
                    ts = sig[team]
                    z = ts.zone_frac(t - A.LAG_HI, t - A.LAG_LO) if (A.AUTO_ZONE_GATE or per != "auto") else 1.0
                    s_ = float((A.kernel(t - ts.shots) * ts.shot_w).sum()) if len(ts.shots) else 0.0
                    vis = ts.vis_frac(t - A.LAG_HI, t - A.LAG_LO)
                    p.append((z * s_, z * A.PRIOR * vis, z * A.HIDDEN * (1 - vis)))
                p = np.array(p)
                if p.sum() > 0:
                    src += n * p / p.sum()
        ide = r["identity"][al]
        for i, t in enumerate(tm):
            segs = [e for e in ide.values() if e["team"] == t]
            dur = sum(e["t1"] - e["t0"] for e in segs) or 1e-9
            contra = sum(e["t1"] - e["t0"] for e in segs if sum(e["ocr"]) >= 1.5 and max(e["ocr"]) / sum(e["ocr"]) >= 0.7
                         and int(np.argmax(e["ocr"])) != i)
            noread = sum(e["t1"] - e["t0"] for e in segs if sum(e["ocr"]) == 0)
            each = [x.n for x in bym[k].get(t, [])]
            sh = [x for x in r["shots"] if x["team"] == t and x["toward"]]
            rows.append({"match": k, "team": t, "al": al, "res": float(v[i] / v.sum() - s[i] / s.sum()),
                         "video": float(v[i]), "scout": float(s[i]), "scout_each": each,
                         "launch_box": sum(1 for x in sh if x["how"] == "box"), "launch_proj": sum(1 for x in sh if x["how"] != "box"),
                         "src_launch_prior_hidden": np.round(src[i] / max(src[i].sum(), 1e-9), 2).tolist(),
                         "vis": round(float(sig[t].vis_v.mean()) if len(sig[t].vis_v) else 0.0, 2),
                         "contra_time": round(contra / dur, 2), "noread_time": round(noread / dur, 2),
                         "calib_views": sorted(r["calibration"])})
rows.sort(key=lambda x: -abs(x["res"]))
res = np.array([x["res"] for x in rows])
print(f"{len(rows)} scouted team-matches; |share error| mean {np.mean(np.abs(res)):.3f}, median {np.median(np.abs(res)):.3f}, "
      f"> 0.15: {np.mean(np.abs(res) > 0.15) * 100:.0f} %")
for x in rows[:16]:
    print(f"  {x['match']:13s} {x['team']:5d} {x['al']:4s} res {x['res']:+.2f}  video {x['video']:4.0f} scout {x['scout']:4.0f} "
          f"{x['scout_each']}  launches box/proj {x['launch_box']}/{x['launch_proj']}  src L/P/H {x['src_launch_prior_hidden']}  "
          f"vis {x['vis']}  contra {x['contra_time']} noread {x['noread_time']}")
# what distinguishes big errors?
big = [x for x in rows if abs(x["res"]) > 0.15]; small = [x for x in rows if abs(x["res"]) <= 0.08]
for col in ("vis", "contra_time", "noread_time"):
    print(f"   {col:12s} big-error median {np.median([x[col] for x in big]):.2f} vs small-error {np.median([x[col] for x in small]):.2f}")
print(f"   hidden share of credit: big {np.median([x['src_launch_prior_hidden'][2] for x in big]):.2f} vs small "
      f"{np.median([x['src_launch_prior_hidden'][2] for x in small]):.2f}")
sd = [abs(x["scout_each"][0] - x["scout_each"][1]) / max(np.mean(x["scout_each"][:2]), 1) for x in big if len(x["scout_each"]) >= 2]
sd2 = [abs(x["scout_each"][0] - x["scout_each"][1]) / max(np.mean(x["scout_each"][:2]), 1) for x in small if len(x["scout_each"]) >= 2]
print(f"   scout disagreement (|a-b|/mean): big {np.median(sd):.2f} vs small {np.median(sd2):.2f}")
json.dump(rows, open(AUDIT / "c10_worst.json", "w"), indent=1, default=float)
