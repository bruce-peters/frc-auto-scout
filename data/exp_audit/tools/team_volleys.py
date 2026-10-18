"""Where did a team's volleys go? For each scouted match of TEAM, every scoreboard jump of its alliance:
the scouted share (scout taps kernel-weighted around the jump, as in score_scouting.py) vs the credited
share (fuel_log), plus the launch evidence each robot had near the jump. Lists the jumps where the
scouts say TEAM shot most of the balls but the credit went elsewhere.

  .venv/Scripts/python data/exp_audit/tools/team_volleys.py 254 [--out output] [--min_share 0.6]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from autoscout import attribution as A, scouting, tba  # noqa: E402
from score_scouting import to_mt_fn, tri  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("team", type=int)
ap.add_argument("--out", default=str(ROOT / "output"))
ap.add_argument("--min_share", type=float, default=0.6)
ap.add_argument("--min_balls", type=int, default=4)
a = ap.parse_args()

bm = scouting.by_match(scouting.load())
tot = {"jumps": 0, "balls": 0, "scout_balls": 0.0, "credit_balls": 0.0, "lost": 0.0}
lost_rows = []
for key in sorted(bm):
    rp = Path(a.out) / key / "result.json"
    if not rp.exists() or a.team not in bm[key]:
        continue
    r = json.loads(rp.read_text())
    info = tba.match(key)
    al = "red" if a.team in info.red else "blue"
    ts = info.teams(al)
    timed = {t: [x for x in bm[key].get(t, []) if x.time_valid] for t in ts}
    if not all(timed.values()):
        print(key, "skipped: scouting timestamps unusable for", [t for t in ts if not timed[t]])
        continue
    tl = r["timeline"]
    to_mt = to_mt_fn(tl)
    a0 = tl.get("auto_start") or 6.13
    log = r["fuel_log"]
    shots = [s for s in r["shots"] if s["alliance"] == al]
    prev = 0
    for vt, c in tl["counts"].get(al, []):
        if c <= prev:
            continue
        k, prev = c - prev, c
        st_c = vt - a0
        w = np.array([np.mean([tri(x.shots - st_c).sum() for x in timed[t]]) for t in ts])
        if w.sum() <= 0:
            continue
        sshare = w / w.sum()
        mt_log = to_mt(vt - 2.0)
        got = np.array([sum(b for m, tm, b in log if tm == t and abs(m - mt_log) < 0.06) for t in ts])
        if got.sum() <= 0:
            continue
        cshare = got / got.sum()
        i = ts.index(a.team)
        tot["jumps"] += 1
        tot["balls"] += k
        tot["scout_balls"] += k * sshare[i]
        tot["credit_balls"] += k * cshare[i]
        if k >= a.min_balls and sshare[i] >= a.min_share and cshare[i] < sshare[i] - 0.3:
            # launch evidence near the jump (kernel-weighted, as split_increments sees it)
            ev = {t: round(float(sum(s["conf"] * A.kernel(np.array([vt - s["t"]]))[0] for s in shots if s["team"] == t)), 2)
                  for t in ts}
            n_near = {t: sum(1 for s in shots if s["team"] == t and A.LAG_LO <= vt - s["t"] <= A.LAG_HI) for t in ts}
            lost = k * (sshare[i] - cshare[i])
            tot["lost"] += lost
            lost_rows.append((key, round(vt, 1), round(mt_log, 1), k, dict(zip(ts, sshare.round(2))),
                              dict(zip(ts, cshare.round(2))), ev, n_near, round(lost, 1)))

print(f"\nteam {a.team}: {tot['jumps']} jumps, {tot['balls']} balls on its alliance's counter;"
      f" scouts give it {tot['scout_balls']:.0f}, credit gives it {tot['credit_balls']:.0f}")
print(f"jumps where scouts say >= {a.min_share:.0%} {a.team} but credit is > 0.3 lower: {len(lost_rows)},"
      f" {tot['lost']:.0f} balls lost\n")
print("match  video_t  match_t  balls  scouted share  credited share  launch evidence (kernel)  launches in window  lost")
for row in sorted(lost_rows, key=lambda x: -x[-1]):
    print(*row)
