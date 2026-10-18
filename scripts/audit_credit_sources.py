"""(Visual audit tool.) Where does each ball's credit come from? Decompose split_increments' weights into
detected-launch evidence, the PRIOR, and the HIDDEN pseudo-shots, over many matches.
Read-only: analyses cached scans in memory, writes nothing under output/."""
import json, pickle, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\bruce\Documents\Coding\auto-scout")
sys.path.insert(0, str(ROOT))
from autoscout import attribution as A, config, pipeline  # noqa: E402

REC = []


def split_rec(increments, teams, to_match_time, log=None):
    for t, n in increments:
        if n <= 0:
            continue
        parts = []
        for ts in teams:
            z = ts.zone_frac(t - A.LAG_HI, t - A.LAG_LO)
            s = float((A.kernel(t - ts.shots) * ts.shot_w).sum()) if len(ts.shots) else 0.0
            vis = ts.vis_frac(t - A.LAG_HI, t - A.LAG_LO)
            parts.append((ts.team, z * s, z * A.PRIOR * vis, z * A.HIDDEN * (1 - vis), z, vis, s))
        REC.append((CUR, t, n, parts))
    return ORIG(increments, teams, to_match_time, log)


ORIG = A.split_increments
A.split_increments = split_rec
CUR = None
keys = sys.argv[1:] or sorted(p.parent.name for p in config.OUTPUT.glob("*/scan.pkl"))[::4]
how = Counter()
for k in keys:
    CUR = k
    with open(config.OUTPUT / k / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    res = pipeline.analyse(md, pipeline.Options())
    for s in res["shots"]:
        how[(s["how"], s["toward"])] += 1
    print("analysed", k, flush=True)

tot = Counter()
pure_fallback = 0
conc = []
evid_balls = 0
for k, t, n, parts in REC:
    w = np.array([p[1] + p[2] + p[3] for p in parts])
    if w.sum() <= 1e-9:
        tot["no weight (even split)"] += n
        pure_fallback += n
        continue
    sh = np.array([p[1] for p in parts]).sum() / w.sum()
    pr = np.array([p[2] for p in parts]).sum() / w.sum()
    hd = np.array([p[3] for p in parts]).sum() / w.sum()
    tot["launch evidence"] += n * sh
    tot["prior"] += n * pr
    tot["hidden pseudo-shots"] += n * hd
    if sh < 1e-9:
        pure_fallback += n
    conc.append((n, (w / w.sum()).max()))
balls = sum(n for _, _, n, _ in REC)
print(f"\n{len(keys)} matches, {len(REC)} counter increments, {balls:.0f} balls")
for kk, v in tot.most_common():
    print(f"  {kk:24s} {100 * v / balls:5.1f} % of balls")
print(f"  increments with NO launch evidence at all: {100 * pure_fallback / balls:.1f} % of balls")
cw = np.array(conc)
print(f"  ball-weighted mean top-robot share per increment: {np.average(cw[:, 1], weights=cw[:, 0]):.2f}"
      f" (1.0 = one robot gets it all, 0.33 = even split)")
print("\nlinked launches by method / toward own hub:")
for kk, v in sorted(how.items(), key=lambda x: -x[1]):
    print(f"  {str(kk):28s} {v}")
json.dump({"tot": tot, "balls": balls, "pure": pure_fallback}, open(Path(__file__).with_suffix(".json"), "w"))
