"""Fit the chosen credit model on every scouted Chezy alliance and write the JSON that
autoscout.attribution.ATTR_MODEL points at.

  .venv/Scripts/python data/exp_attrmodel/tools/export_model.py DS.pkl OUT.json [--spec lin/all+dist+mlprior]
        [--lam 3e-4] [--cv CV.json]   (--cv: copy the cross-validated metrics of that spec into the file)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import fit_cv as FC  # noqa: E402

# shooting terms (data/exp_launch/RESULTS.md, prod_signals.py): spec extra -> (model key, per-jump form)
SHOOT = {"fx:dinoc": ("g_dinoc", "window"), "fx:drainraw": ("g_drain", "window"),
         "fx:dinock": ("g_dinoc", "kernel"), "fx:drainrawk": ("g_drain", "kernel")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds")
    ap.add_argument("out")
    ap.add_argument("--spec", default="lin/all+dist+mlprior")
    ap.add_argument("--lam", type=float, default=3e-4)
    ap.add_argument("--cv")
    a = ap.parse_args()
    feats, recs = FC.load(a.ds)
    sc = [r for r in recs if r.get("overlay") and "scout_n" in r and r.get("tba_total") and r["video_share"] is not None
          and r["scout_n"].sum() > 0 and len(r["n"]) > 0]
    spec = FC.parse_spec(a.spec)
    assert spec.form == "lin" and spec.sset == "all" and not spec.priors and \
        set(spec.extras) <= {"dist", "mlprior"} | set(SHOOT), \
        "attribution.py implements lin/all[+dist][+mlprior][+fx:dinoc|fx:dinock][+fx:drainraw|fx:drainrawk] only"
    shoot = [e for e in spec.extras if e in SHOOT]
    forms = {SHOOT[e][1] for e in shoot}
    keys_ = [SHOOT[e][0] for e in shoot]
    assert len(forms) <= 1 and len(set(keys_)) == len(keys_), "one form (window or kernel) and one term per signal"
    th = FC.fit(spec, FC.Pack(sc, feats), a.lam)
    m_in = FC.metrics(FC.predict(spec, th, FC.Pack(sc, feats)), sc)
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
        "doc": ("w_r = (zone_r + zone_eps)^gz * (s_r + prior * vis_r + hidden * (1 - vis_r))^b * "
                "exp(g_dist * (dist_r - dist_center) / dist_scale), normalised over the alliance's robots for each "
                "scoreboard jump (dist_r: mean distance to its hub over the window, term dropped if unknown); then each "
                "team's match total is multiplied by opr_share_r^g_opr (opr_share floored at opr_floor) and the "
                "alliance renormalised to the same total. data/exp_attrmodel/RESULTS_section.md"),
        "fit": {"spec": a.spec, "lambda": a.lam, "alliances": len(sc), "matches": len({r["key"] for r in sc}),
                "source": str(Path(a.ds).name), "in_sample": m_in, "theta": th},
    }
    if shoot:
        import hashlib
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "exp_launch"))
        import prod_signals as PS
        ridge = Path(__file__).resolve().parents[3] / "models" / "shooting_ridge.npz"
        model["version"] = 2  # needs the shooting factors at runtime
        model["g_dinoc"], model["g_drain"] = 0.0, 0.0
        for e in shoot:
            model[SHOOT[e][0]] = float(th["g_" + e[3:]])
        model["shooting"] = {
            "form": forms.pop(), "ridge": "models/shooting_ridge.npz",
            "ridge_sha256": hashlib.sha256(ridge.read_bytes()).hexdigest() if ridge.exists() else None,
            "embedding": "+".join(PS.FEATS), "center": "per robot, per view, within the match",
            "grid_step": PS.GRID_STEP, "dino_shift": PS.DINO_SHIFT, "drain_shift": PS.DRAIN_SHIFT,
            "drain_sub": PS.DRAIN_SUB, "drain_gap": PS.DRAIN_GAP, "code": "data/exp_launch/prod_signals.py"}
        model["doc"] += (" Shooting (version 2): w_r is also multiplied by (1 + S_dinoc_r)^g_dinoc * (1 + S_drain_r)^g_drain"
                         " before the per-jump normalisation; S_x_r for shooting.form window = the mean of the robot's"
                         " prod_signals series over the grid points in [t - LAG_HI, t - LAG_LO] (0 if none), for kernel ="
                         " sum(kernel(t - grid) * value); 0 when the series is missing (no embeddings: factor 1).")
    if a.cv:
        cv = json.loads(Path(a.cv).read_text())
        mm = cv["models"].get(spec.name())
        if mm:
            model["fit"]["cv_lomo"] = {k: mm[k] for k in ("tv", "corr", "top", "fuel_mae", "fuel_corr", "slope", "n")}
        model["fit"]["cv_blend_0.7_0.3"] = cv["baselines"]["0.7 video + 0.3 OPR"]
        model["fit"]["cv_video_only"] = cv["baselines"]["video only (stored)"]
    Path(a.out).write_text(json.dumps(model, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in model.items() if k not in ("fit", "doc")}, indent=1))
    print("in-sample:", FC.fmt(m_in))


if __name__ == "__main__":
    main()
