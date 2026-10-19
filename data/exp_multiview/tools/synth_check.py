"""Synthetic check of autoscout/triangulate.py on real cameras: a drag-free ball projected through a match's
blue_end and main CameraTracks (cv2, with distortion), on the scan's frame grid, then paired and fitted back.

  .venv/Scripts/python data/exp_multiview/tools/synth_check.py [match] [--noise PX]
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from autoscout import config, fuel, triangulate as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("key", nargs="?", default="2026cc_qm24")
ap.add_argument("--noise", type=float, default=0.0)
a = ap.parse_args()
with open(config.OUTPUT / a.key / "scan.pkl", "rb") as fh:
    md = pickle.loads(fh.read())
grid = np.unique(np.concatenate([np.array(tr.pts)[:, 0] for tr in md.tracks_all["main"]]))
ts = grid[np.searchsorted(grid, 80.0):][:25]
rng = np.random.default_rng(0)
for x_true in (np.array([3.0, 2.5, 0.6, 2.0, 1.5, 7.0]), np.array([2.0, 5.5, 0.55, 2.5, -1.5, 6.5])):
    X = T.path(x_true, ts - ts[0])
    fake = {}
    for v in ("main", "blue_end"):
        cam = md.cams[v]
        uv = np.array([cam.at(t).project(X[n:n + 1])[0] for n, t in enumerate(ts)]) + rng.normal(0, a.noise, (len(ts), 2))
        fake[v] = [fuel.FlightTrack(0, [(t, u, w, 10.0) for t, (u, w) in zip(ts, uv)])]
    mv = T.MultiView(md, tracks_all=fake)
    f = mv.pairs_for("blue_end", 0)
    if f is None:
        print("no pair accepted")
        continue
    lp = f["launch"]
    s_true = T._launch_s(x_true, T.LAUNCH_Z)
    xy_true = x_true[:2] + x_true[3:5] * s_true
    print(f"true launch {np.round(xy_true, 3)}  fitted {np.round(lp['xy'], 3)}  error {np.hypot(*(lp['xy'] - xy_true)):.3f} m"
          f"  claimed sd {np.sqrt(np.trace(lp['cov'])):.3f} m  rms px {np.round(f['rms'], 3)}")
