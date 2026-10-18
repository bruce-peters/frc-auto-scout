"""Dump every installed scan's camera calibration (K, k1, camera centre, rms, poses) per view.

Read-only on output/. Writes data/exp_calib/cams_output.json. Used to look at the per-event spread of
end-camera intrinsics / centres before building the event prior.
"""
import gc
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(r"C:\Users\bruce\Documents\Coding\auto-scout")
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from autoscout import field as F  # noqa: E402,F401

src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "output"
out = {}
for p in sorted(src.glob("*/scan.pkl")):
    with open(p, "rb") as fh:
        md = pickle.load(fh)
    row = {"event": md.info.event_key, "prod": md.prod, "views": {}}
    for v, c in md.cams.items():
        ts = list(getattr(c, "times", [0.0]))
        cen = np.array([c.at(t).center_world() for t in ts])
        row["views"][v] = {"f": float(c.K[0][0]), "cx": float(c.K[0][2]), "cy": float(c.K[1][2]),
                           "k1": float(c.dist[0]), "centre": np.median(cen, 0).round(3).tolist(),
                           "centre_spread": cen.std(0).round(3).tolist(), "rms": float(c.rms), "n_tags": int(c.n_tags),
                           "poses": len(ts), "size": list(c.size), "type": type(c).__name__}
    out[p.parent.name] = row
    print(p.parent.name, md.prod, {v: (round(x["f"]), round(x["rms"], 2), x["centre"]) for v, x in row["views"].items()},
          flush=True)
    del md
    gc.collect()
(ROOT / "data" / "exp_calib" / "cams_output.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
