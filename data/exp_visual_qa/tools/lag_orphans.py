"""How many counted balls land on counter increments with (almost) no launch evidence inside the credit kernel
(attribution.kernel, LAG_LO..LAG_HI) but with many launches 5.5-10 s earlier (a long dump still queuing in the hub)?
Those balls are split by the PRIOR / HIDDEN terms instead of by who launched them.

  .venv/Scripts/python data/exp_visual_qa/tools/lag_orphans.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from autoscout import attribution as A  # noqa: E402

C = ROOT / "data/exp_visual_qa/cache"
tot = orphan = late = 0
for d in sorted(C.iterdir()):
    if not (d / "result.json").exists():
        continue
    r = json.loads((d / "result.json").read_text())
    for al in ("red", "blue"):
        st = np.array([s["t"] for s in r["shots"] if s["alliance"] == al and s["team"] is not None])
        sw = np.array([s["conf"] for s in r["shots"] if s["alliance"] == al and s["team"] is not None])
        prev = 0
        for vt, c in r["timeline"]["counts"].get(al, []):
            if c <= prev:
                continue
            k, prev = c - prev, c
            tot += k
            ev = float((A.kernel(vt - st) * sw).sum()) if len(st) else 0.0
            if ev < 1.0:
                orphan += k
                n_late = int((((vt - st) > A.LAG_HI) & ((vt - st) <= 10.0)).sum()) if len(st) else 0
                if n_late >= 10:
                    late += k
print(f"counted balls {tot}; on increments with kernel launch evidence < 1: {orphan} ({100 * orphan / tot:.1f} %);"
      f" of those with >= 10 launches 5.5-10 s earlier (dump still queuing): {late} ({100 * late / tot:.1f} % of all)")
