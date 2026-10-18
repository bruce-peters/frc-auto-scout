"""Cycle 11: which scouted shots does the video miss, by camera coverage and robot position?

Scouted shots (time-valid rows, tap - 0.8 s, timer = auto_start + t) of every scouted robot. A shot is
'detected' if a credited toward-hub launch of that team lies within +-1.0 s. For each shot: is the robot
in zone / visible, which views have a box of that team within 0.15 s (dets.json team column), its distance
to its own hub and to its own end camera. Recall by those factors (ball-weighted, scouts averaged).
Usage: c11_recall_views.py [match ...]   (default: the 14 original cc matches)
"""
import json
import sys
from collections import defaultdict

import numpy as np

from common import AUDIT, DIRS, F, P, TL, load, match_keys, path_video_times, tba, time_valid
from autoscout import scouting

bym = scouting.by_match(scouting.load())
keys = sys.argv[1:] or sorted(set(match_keys("v1")) & {k for k in match_keys("v2") if k.startswith("2026cc_qm")})
TAP = 0.8
acc = defaultdict(lambda: np.zeros(2))  # factor -> (balls, detected balls)
for k in keys:
    r = load(k, "v2"); info = tba.match(k); tl = TL(r)
    dets = json.loads((DIRS["v2"] / k / "dets.json").read_text())
    by_team = defaultdict(list)
    for d in dets:
        if d[7] is not None:
            by_team[d[7]].append((d[0], d[1]))
    for t in by_team:
        by_team[t].sort()
    cams = r["calibration"]
    for al in ("red", "blue"):
        hub = F.hub_center(al)
        for team in info.teams(al):
            rows = [x for x in bym.get(k, {}).get(team, []) if time_valid(x)]
            if not rows:
                continue
            L = np.array([s["t"] for s in r["shots"] if s["team"] == team and s["toward"]])
            pts = path_video_times(r["paths"].get(str(team), []), tl)
            bt = np.array([x[0] for x in by_team.get(team, [])]); bv = [x[1] for x in by_team.get(team, [])]
            w = 1.0 / len(rows)
            for row in rows:
                for st in tl.auto_start + row.shots - TAP:
                    det = bool(len(L) and np.min(np.abs(L - st)) <= 1.0)
                    p = P._track_position(pts, st) if len(pts) else None
                    seen = set()
                    if len(bt):
                        j0, j1 = np.searchsorted(bt, st - 0.15), np.searchsorted(bt, st + 0.15)
                        seen = {bv[j] for j in range(j0, j1)}
                    own_end = f"{al}_end"
                    cov = ("main+" if "main" in seen else "") + ("own_end" if own_end in seen else "") + \
                          ("+opp_end" if (("blue_end" if al == "red" else "red_end") in seen) else "")
                    cov = cov.strip("+") or "no box"
                    facs = [("all", "all"), ("boxes in", cov)]
                    if p is not None:
                        dh = float(np.hypot(p[0] - hub[0], p[1] - hub[1]))
                        facs.append(("dist to own hub", f"{int(dh // 1.5) * 1.5:.1f}-{int(dh // 1.5) * 1.5 + 1.5:.1f} m"))
                        zone = F.in_alliance_zone(al, p[0])
                        facs.append(("zone", "in" if zone else "out"))
                        facs.append(("observed fix", "yes" if p[2] else "interpolated"))
                        facs.append(("side (y)", "near y<2.7" if p[1] < 2.7 else ("far y>5.4" if p[1] > 5.4 else "middle")))
                    else:
                        facs.append(("zone", "position unknown"))
                    mt = tl.match_time(st)
                    facs.append(("period", "auto" if mt is not None and mt < 20 else "teleop"))
                    for f in facs:
                        acc[f] += (w, w * det)
print(f"{len(keys)} matches: scouted balls and detected share (launch of that team within +-1 s)")
cur = None
for (g, v), (n, d) in sorted(acc.items()):
    if g != cur:
        print(f"  {g}:")
        cur = g
    print(f"     {v:24s} balls {n:7.0f}  detected {100 * d / n:5.1f} %")
