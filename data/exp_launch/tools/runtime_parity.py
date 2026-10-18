"""The runtime rule for _model_weights (version-2 model JSON) vs fit_cv's logw, per scoreboard jump.

Runtime rule (numpy, attribution._model_weights + the shooting terms), per robot r at jump t:
  e   = s_r + (prior * vis_r * (1 - prior_off_r) + hidden * (1 - vis_r) * hid_on_r) * pres_r
  lw  = b * log(e + 1e-9) + gz * log(zone_r + zone_eps) + g_dist * dz_r
        + g_dinoc * log1p(S_dinoc_r) + g_drain * log1p(S_drain_r)
  w   = softmax over the alliance's robots
with S_x_r = s_dinoc / s_drainraw (window form). Compared with fit_cv.Spec.logw at the exported theta.

  runtime_parity.py DS.pkl MODEL.json key [key ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"C:\Users\bruce\Documents\Coding\auto-scout")
sys.path.insert(0, str(ROOT / "data" / "exp_attrmodel" / "tools"))
import fit_cv as FC  # noqa: E402

ds, mj, keys = sys.argv[1], json.loads(Path(sys.argv[2]).read_text()), sys.argv[3:]
feats, recs = FC.load(ds)
spec = FC.parse_spec(mj["fit"]["spec"])
th = {k: torch.tensor(v) for k, v in mj["fit"]["theta"].items()}
fi = {f: i for i, f in enumerate(feats)}
for k in keys:
    rs = [r for r in recs if r["key"] == k and r.get("overlay") and len(r["n"])]
    P = FC.Pack(rs, feats)
    with torch.no_grad():
        ref = torch.softmax(spec.logw(th, P), 1).numpy()
    mine = []
    for r in rs:
        X = np.asarray(r["X"])
        poff = np.asarray(r.get("prior_off", np.zeros(3)), float)
        for j in range(len(X)):
            f = lambda n: X[j, :, fi[n]]
            vis = f("vis")
            e = f("s_all") + (mj["prior"] * vis * (1 - poff) + mj["hidden"] * (1 - vis) * f("hid_on")) * f("pres")
            d = f("dist")
            dz = np.where(np.isnan(d), 0.0, (d - mj["dist_center"]) / mj["dist_scale"])
            lw = mj["b"] * np.log(e + 1e-9) + mj["gz"] * np.log(f("zone") + mj["zone_eps"]) + mj["g_dist"] * dz
            if mj.get("version", 1) >= 2:
                sfx = "" if mj["shooting"]["form"] == "window" else "k"
                lw = lw + mj["g_dinoc"] * np.log1p(f("s_dinoc" + sfx)) + mj["g_drain"] * np.log1p(f("s_drainraw" + sfx))
            w = np.exp(lw - lw.max())
            mine.append(w / w.sum())
    mine = np.array(mine)
    print(f"{k}: {len(mine)} jumps x 3 robots, max |w_runtime - w_fit_cv| {np.abs(mine - ref).max():.2e}")
