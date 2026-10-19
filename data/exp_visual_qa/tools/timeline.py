"""Per-robot timeline of one alliance over a match: scout taps (per scout, shifted -0.8 s to launch time) vs the
system's launches per credited team, in 2 s bins, plus the alliance counter. Shows whether a scout's taps follow a
different robot's launches (scout watching the wrong robot) or the credit's.

  .venv/Scripts/python data/exp_visual_qa/tools/timeline.py MATCH ALLIANCE [--t0 0 --t1 200 --bin 2]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from autoscout import scouting, tba  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("match")
ap.add_argument("alliance")
ap.add_argument("--t0", type=float, default=0)
ap.add_argument("--t1", type=float, default=200)
ap.add_argument("--bin", type=float, default=2.0)
a = ap.parse_args()
r = json.loads((ROOT / "data/exp_visual_qa/cache" / a.match / "result.json").read_text())
info = tba.match(a.match)
ts = info.teams(a.alliance)
bm = scouting.by_match(scouting.load())
a0 = r["timeline"].get("auto_start") or 6.13
edges = np.arange(a.t0, a.t1 + a.bin, a.bin)
cols, names = [], []
for t in ts:
    for i, row in enumerate(bm[a.match].get(t, [])):
        cols.append(np.histogram(row.shots + a0 - 0.8, edges)[0])
        names.append(f"S{t}.{i}{'' if row.time_valid else '!'}")
for t in ts:
    lt = [s["t"] for s in r["shots"] if s["team"] == t]
    cols.append(np.histogram(lt, edges)[0])
    names.append(f"L{t}")
lt = [s["t"] for s in r["shots"] if s["alliance"] == a.alliance and s["team"] is None]
cols.append(np.histogram(lt, edges)[0])
names.append("Lnone")
cnt = r["timeline"]["counts"][a.alliance]
inc = np.zeros(len(edges) - 1)
prev = 0
for vt, c in cnt:
    if c > prev:
        j = np.searchsorted(edges, vt - 2.4) - 1  # counter time -> ~launch time
        if 0 <= j < len(inc):
            inc[j] += c - prev
        prev = c
cols.append(inc.astype(int))
names.append("ctr-2.4")
fl = {t: np.zeros(len(edges) - 1) for t in ts}
print(f"{a.match} {a.alliance} teams {ts}; scout totals " +
      ", ".join(f"{t}: {[len(x.shots) for x in bm[a.match].get(t, [])]}" for t in ts) +
      "; credited " + ", ".join(f"{t}: {r['teams'][str(t)]['fuel']}" for t in ts))
print("t(launch)  " + " ".join(f"{n:>9}" for n in names))
for j in range(len(edges) - 1):
    row = [c[j] for c in cols]
    if sum(row) == 0:
        continue
    print(f"{edges[j]:7.1f}    " + " ".join(f"{x:9d}" for x in row))
