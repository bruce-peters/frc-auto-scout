"""Do robots that lose most of their labelled time under a config (duplicate labels allowed) really score less?

For the scouted Chezy quals: team-matches whose labelled-time coverage drops below 50 % under config B but not
under A, with their scouted shot share in the alliance (a robot absorbed by a teammate's label that really
shot a lot would be a false merge).
  coverage_check.py full1 base d2all2
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from autoscout import scouting, tba  # noqa: E402

tag, a, b = sys.argv[1:4]
E = json.loads((ROOT / "data" / "exp_dino" / "eval" / f"{tag}.json").read_text())["per_match"]
bm = scouting.by_match(scouting.load())
lost, kept_low, all_share = [], [], []
for key, teams in bm.items():
    if key not in E[a] or "cov" not in E[a][key]:
        continue
    info = tba.match(key)
    for al in ("red", "blue"):
        ts = info.teams(al)
        if not all(teams.get(t) for t in ts):
            continue
        n = np.array([np.mean([r.n for r in teams[t]]) for t in ts])
        if n.sum() <= 0:
            continue
        sh = n / n.sum()
        for t, s in zip(ts, sh):
            ca, cb = E[a][key]["cov"][str(t)], E[b][key]["cov"][str(t)]
            all_share.append(s)
            if cb < 0.5 <= ca:
                lost.append((key, t, round(ca, 2), round(cb, 2), round(float(s), 2)))
            elif cb < 0.5 and ca < 0.5:
                kept_low.append(s)
print(f"team-matches: {len(all_share)}; coverage < 50 % under {b} but not {a}: {len(lost)}; "
      f"their mean scouted shot share {np.mean([x[4] for x in lost]) if lost else float('nan'):.2f} "
      f"(all robots {np.mean(all_share):.2f}; low under both: {len(kept_low)}, share {np.mean(kept_low) if kept_low else float('nan'):.2f})")
for x in sorted(lost, key=lambda x: -x[4]):
    print("  ", x)
