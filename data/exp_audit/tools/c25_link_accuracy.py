"""Cycle 25: per-link-type accuracy of credited launches, judged by the scouts.

For every credited toward-hub launch in a scouted cc qual (all three robots of its alliance scouted,
time_valid rows): which alliance robots have a scouted shot within +-1 s. Judged when exactly one does.
Accuracy = credited team == that robot, by link type (box unambiguous / box ambiguous / proj / proj_hidden),
view and period. Then: down-weighting the weakest link types (conf x f) -> scout share corr / OPR.
"""
import copy
import json
from collections import Counter, defaultdict

import numpy as np

from common import AUDIT, DIRS, TL, load, match_keys, resplit, scout_metrics, share_metrics, tba, time_valid
from autoscout import scouting, views

bym = scouting.by_match(scouting.load())
acc = defaultdict(lambda: np.zeros(2))
R, INFO, AMBI = {}, {}, {}
for k in match_keys("v2"):
    r = load(k, "v2"); info = tba.match(k)
    R[k], INFO[k] = r, info
    # ambiguity of box links (>= 2 team boxes within the link distance), as in c24
    prod = "chezy" if r["timeline"]["layout"] == "chezy" else "first"
    vl = {v.name: v for v in views.PRODUCTIONS[prod]}
    rows = json.loads((DIRS["v2"] / k / "dets.json").read_text())
    by_vt = defaultdict(lambda: defaultdict(list))
    for d in rows:
        by_vt[d[1]][d[0]].append(d)
    tk = {v: np.array(sorted(x)) for v, x in by_vt.items()}
    amb = set()
    for si, s in enumerate(r["shots"]):
        if s["how"] != "box" or s["team"] is None or s["view"] not in tk:
            continue
        ts = tk[s["view"]]; j = int(np.argmin(np.abs(ts - s["t"])))
        ox, oy = vl[s["view"]].offset(1920, 1080)
        n = set()
        for d in by_vt[s["view"]][ts[j]]:
            if d[7] is None:
                continue
            x0, y0, x1, y1 = d[2] - ox, d[3] - oy, d[4] - ox, d[5] - oy
            bw, bh = x1 - x0, y1 - y0
            dx = max(x0 - 0.25 * bw - s["u"], 0, s["u"] - x1 - 0.25 * bw); dy = max(y0 - 0.9 * bh - s["v"], 0, s["v"] - y1)
            if np.hypot(dx, dy) / max(bh, 1) <= 0.6:
                n.add(d[7])
        if len(n) >= 2:
            amb.add(si)
    AMBI[k] = amb
    if k not in bym:
        continue
    tl = TL(r)
    for al in ("red", "blue"):
        tm = info.teams(al)
        rows_ = {t: [x for x in bym[k].get(t, []) if time_valid(x)] for t in tm}
        if not all(rows_.values()):
            continue
        st = {t: np.sort(np.concatenate([tl.auto_start + x.shots - 0.8 for x in rows_[t]])) for t in tm}
        for si, s in enumerate(r["shots"]):
            if s["team"] not in tm or not s["toward"]:
                continue
            sh = [t for t in tm if len(st[t]) and np.min(np.abs(st[t] - s["t"])) <= 1.0]
            if len(sh) != 1:
                continue
            how = s["how"] if s["how"] != "box" else ("box ambiguous" if si in amb else "box unambiguous")
            mt = tl.match_time(s["t"])
            for key in (("link", how), ("view", s["view"]), ("period", "auto" if mt is not None and mt < 20 else "teleop"),
                        ("all", "all")):
                acc[key] += (1, int(s["team"] == sh[0]))
print("credited launches judged by the scouts (exactly one alliance robot scouted shooting within +-1 s):")
for (g, v), (n, ok) in sorted(acc.items()):
    print(f"   {g:7s} {v:16s} n {n:5.0f}  correct {100 * ok / n:5.1f} %")
print("   chance ~33 %")

res = {}
for name, f_proj, f_amb in (("base", 1.0, 1.0), ("proj x0.5", 0.5, 1.0), ("proj_hidden x0.5", None, 1.0), ("box-ambiguous x0.5", 1.0, 0.5)):
    pm = {}
    for k in R:
        r2 = copy.deepcopy(R[k])
        for si, s in enumerate(r2["shots"]):
            if name == "proj x0.5" and s["how"] == "proj":
                s["conf"] *= f_proj
            elif name == "proj_hidden x0.5" and s["how"] == "proj_hidden":
                s["conf"] *= 0.5
            elif name == "box-ambiguous x0.5" and si in AMBI[k]:
                s["conf"] *= f_amb
        pm[k] = resplit(r2, INFO[k])
    print(f"{name:20s}", share_metrics(pm), scout_metrics(pm), flush=True)
