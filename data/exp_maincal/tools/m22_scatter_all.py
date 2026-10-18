"""centre_scatter of the stored main camera of every scan in output/ (read only): which views would the new main
path touch? Also |rvec| range and rvec-branch flips of every stored camera. Writes data/exp_maincal/m22_scatter_all.json."""
import gc
import json
import pickle
import numpy as np
from common import EXP, F, M, ROOT
rows = {}
for p in sorted((ROOT / "output").glob("*/scan.pkl")):
    try:
        with open(p, "rb") as fh:
            md = pickle.load(fh)
    except Exception as e:  # a scan being written
        print(p.parent.name, "unreadable", e)
        continue
    r = {}
    for v, cam in md.cams.items():
        if not hasattr(cam, "rvecs"):
            continue
        rv = np.array(cam.rvecs)
        flips = sum(1 for a, b in zip(rv[:-1], rv[1:]) if np.linalg.norm(a - b) > 0.5 and F._rot_angle(a, b) < 10)
        r[v] = {"scatter": round(M.centre_scatter(cam), 3) if v == "main" else None,
                "rnorm": [round(float(np.linalg.norm(rv, axis=1).min()), 3), round(float(np.linalg.norm(rv, axis=1).max()), 3)],
                "flips": flips, "main_fix": getattr(cam, "main_fix", None)}
    rows[p.parent.name] = r
    m = r.get("main", {})
    print(f"{p.parent.name:18s} main scatter {m.get('scatter')}  flips {[(v, x['flips']) for v, x in r.items() if x['flips']]}", flush=True)
    del md
    gc.collect()
(EXP / "m22_scatter_all.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
