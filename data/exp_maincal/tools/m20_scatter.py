"""How much do the stored (plain) main-camera window poses disagree about the camera centre? Median distance of
the window centres from their median, per match; and the static fit's tag rms vs the plain's.
Usage: m20_scatter.py"""
import json
import numpy as np
from common import EXP, F, M, TAGS, keys, load_extract
rows = {}
ev = {}
for k in keys():
    ex = load_extract(k)
    cam = ex["cams"].get("main")
    if cam is None or not hasattr(cam, "times"):
        continue
    c = np.array([cam.at(t).center_world() for t in cam.times])
    sc = float(np.median(np.linalg.norm(c - np.median(c, 0), axis=1)))
    rows[k] = {"scatter_m": round(sc, 3), "plain_rms": round(cam.rms, 3), "poses": len(cam.times)}
    print(f"{k:18s} centre scatter {sc:6.3f} m  plain rms {cam.rms:.2f}  poses {len(cam.times)}")
(EXP / "m20_scatter.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
