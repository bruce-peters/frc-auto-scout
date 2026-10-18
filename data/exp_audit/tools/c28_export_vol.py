"""Cycle 28: full-data fit of lin/all+dist+mlprior(+volexp), exactly as data/exp_attrmodel/tools/export_model.py
(same alliance filter, same fit), written in its model-file shape plus g_vol and the volume-signal settings.

  c28_export_vol.py DS.pkl LAM OUT.json
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import common  # noqa: F401  (puts the repo root on sys.path for vol_signal)
import fit_cv_vol2 as FC
import vol_signal as VS

ds, lam, out = sys.argv[1], float(sys.argv[2]), Path(sys.argv[3])
feats, recs = FC.load(ds)
sc = [r for r in recs if r.get("overlay") and "scout_n" in r and r.get("tba_total") and r["video_share"] is not None
      and r["scout_n"].sum() > 0 and len(r["n"]) > 0]
res = {}
for txt in ("lin/all+dist+mlprior", "lin/all+dist+mlprior+volexp"):
    spec = FC.parse_spec(txt)
    th = FC.fit(spec, FC.Pack(sc, feats), lam)
    m_in = FC.metrics(FC.predict(spec, th, FC.Pack(sc, feats)), sc)
    res[txt] = (th, m_in)
    print(f"{txt:32s} lambda {lam:g}  theta {json.dumps({k: round(v, 4) for k, v in th.items()})}")
    print(f"{'':32s} in-sample TV {m_in['tv']:.4f}  corr {m_in['corr']:.4f}  fuel MAE {m_in['fuel_mae']:.2f}")
th, m_in = res["lin/all+dist+mlprior+volexp"]
model = {
    "version": 1,
    "form": "lin/all",
    "prior": float(np.exp(th["lp"])),
    "hidden": float(np.exp(th["lh"])),
    "b": float(np.exp(th["lb"])),
    "gz": float(th.get("gz", 1.0)),
    "g_dist": float(th.get("g_dist", 0.0)),
    "dist_center": 3.5,
    "dist_scale": 2.0,
    "g_opr": float(th.get("g_ml", 0.0)),
    "opr_floor": FC.LOG_FLOOR,
    "zone_eps": 1e-3,
    "g_vol": float(th["g_vol"]),
    "vol": {"min_speed": VS.MIN_SPEED, "ball_r": VS.BALL_R,
            "vol_signal_sha256": hashlib.sha256(Path(VS.__file__).read_bytes()).hexdigest()},
    "doc": ("as export_model.py, times (1 + s_beq_r)^g_vol per scoreboard jump before the per-jump normalisation; "
            "s_beq_r = sum(attribution.kernel(t - be_t) * be) over the robot's vol_signal.ball_equivalents rows "
            "(data/exp_audit/tools/vol_signal.py, failure_finder_cycle28.md)"),
    "fit": {"spec": "lin/all+dist+mlprior+volexp", "lambda": lam, "alliances": len(sc),
            "matches": len({r["key"] for r in sc}), "source": Path(ds).name, "in_sample": m_in, "theta": th},
}
out.write_text(json.dumps(model, indent=1), encoding="utf-8")
print("wrote", out)
