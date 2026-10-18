"""Synthetic check of autoscout.arcs.arc_fit on a real camera: simulate a hub shot from one robot, add a decoy
robot on (nearly) the same line of sight, and see whether the fit prefers the true launcher.

  .venv/Scripts/python data/exp_arcs/tools/synth_check.py [match] [view]
"""
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from autoscout import arcs, config, field as F  # noqa: E402

key = sys.argv[1] if len(sys.argv) > 1 else "2026cc_qm24"
view = sys.argv[2] if len(sys.argv) > 2 else "blue_end"
with open(config.OUTPUT / key / "scan.pkl", "rb") as fh:
    raw = fh.read()
md = pickle.loads(raw)
del raw
cam = md.cams[view].at(60.0)
al = "blue" if view in ("blue_end", "main") else "red"
hub = np.array([*F.hub_center(al), config.HUB_OPENING_HEIGHT])
C = cam.center_world()
rng = np.random.default_rng(0)
print(view, "camera at", np.round(C, 2))
for trial in range(12):
    # true launcher somewhere in the alliance zone, decoy along the camera ray through its launch point
    x = rng.uniform(0.8, 3.6) if al == "blue" else rng.uniform(12.9, 15.7)
    y = rng.uniform(0.8, 7.2)
    L = np.array([x, y, arcs.LAUNCH_Z])
    ray = L - C
    f = rng.choice([0.6, 0.75, 1.3, 1.6])
    D = C + f * ray  # same image point, different depth (and height, so put it back at LAUNCH_Z)
    decoy = (D[0], D[1])
    T = rng.uniform(0.8, 1.6)
    for dur in (0.13, 0.27, 0.5):
        s = np.arange(0.05, 0.05 + dur + 1e-9, 1 / 30)
        P = arcs.trajectory(L, hub, T, s)
        uv, _ = arcs._project(cam, P)
        uv = uv + rng.normal(0, 2.0, uv.shape)
        pts = np.column_stack([60.0 + s, uv])
        fits = arcs.arc_fit(pts, cam, [(x, y), decoy], hub)
        print(f"  true ({x:5.2f},{y:5.2f}) decoy ({decoy[0]:5.2f},{decoy[1]:5.2f}) T {T:.2f} dur {dur:.2f}: "
              f"cost true {fits[0]['cost']:7.1f} decoy {fits[1]['cost']:7.1f} | rms {fits[0]['rms']:5.1f} {fits[1]['rms']:5.1f} | "
              f"T fit {fits[0]['T']:.2f} {fits[1]['T']:.2f}")
