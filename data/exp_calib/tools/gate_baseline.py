"""What does the cross-view gate (pipeline.cross_view_error) give for end views that calibrate normally?

For every installed scan (read-only) and each calibrated end view: the median / q75 distance between the
view's robots and the nearest same-alliance main-view robot at the same instant, plus a null baseline
with the main-view positions taken 30 s later (what a camera unrelated to the scene would score).
Also each view's focal length vs its event prior median. Writes data/exp_calib/gate_baseline.json.
Usage: gate_baseline.py [SCAN_DIR]
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from autoscout import field as F, pipeline as P, robots, views  # noqa: E402

src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "output"


from gate_baseline_lib import null_error  # noqa: E402


rows = {}
for p in sorted(src.glob("*/scan.pkl")):
    with open(p, "rb") as fh:
        md = pickle.load(fh)
    key = p.parent.name
    for v in ("blue_end", "red_end"):
        cam = md.cams.get(v)
        if cam is None:
            continue
        g = P.cross_view_error(md, v, cam)
        pr = F.load_calib_prior(md.info.event_key, v, tuple(cam.size), exclude=key)
        rows[f"{key}|{v}"] = {"median": g["median"], "q75": g["q75"], "n": g["n"],
                              "null": null_error(md, v, cam), "rms": cam.rms, "f": cam.K[0][0],
                              "f_ratio": cam.K[0][0] / pr["f"] if pr else None,
                              "recovered": getattr(cam, "recovered", None)}
        r = rows[f"{key}|{v}"]
        print(f"{key:18s} {v:8s} median {r['median']:.2f} q75 {r['q75']:.2f} n {r['n']:5d} null "
              f"{r['null'] if r['null'] is None else round(r['null'], 2)}  f {r['f']:.0f} "
              f"({r['f_ratio'] if r['f_ratio'] is None else round(r['f_ratio'], 2)})", flush=True)
    del md
    gc.collect()
(ROOT / "data" / "exp_calib" / f"gate_baseline_{src.name}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
m = np.array([r["median"] for r in rows.values() if r["n"] >= P.CALIB_GATE_MIN_N])
nl = np.array([r["null"] for r in rows.values() if r["null"] is not None])
fr = np.array([r["f_ratio"] for r in rows.values() if r["f_ratio"] is not None])
print(f"{len(m)} calibrated end views: gate median of medians {np.median(m):.2f} m, "
      f"q90 {np.quantile(m, .9):.2f}, max {m.max():.2f}; below {P.CALIB_GATE_M} m: {np.mean(m < P.CALIB_GATE_M):.0%}")
print(f"null (main view 30 s later): median {np.median(nl):.2f} m, min {nl.min():.2f}, q10 {np.quantile(nl, .1):.2f}")
print(f"focal vs event median: {np.mean(np.abs(fr - 1) > F.PRIOR_F_TOL):.0%} of views beyond {F.PRIOR_F_TOL:.0%} "
      f"({np.sum(np.abs(fr - 1) > F.PRIOR_F_TOL)} of {len(fr)})")
