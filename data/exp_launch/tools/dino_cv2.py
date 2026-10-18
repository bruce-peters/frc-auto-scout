"""Leak-free check of the DINO shooting feature in the credit model (double leave-out).

In cv_launch_lam3e-5.json the DINO feature of a TRAINING match j came from a ridge model trained on every other
scouted match, including the test match m, so m's scout labels reached the credit-model fit via j's features.
Here, for each outer fold m: the training matches' s_dino come from ridge models fitted without {j, m}; the test
match keeps its leave-one-out feature (ridge without m). Same ridge as dino_fit.py (box_inner+box_cls, lam 30,
lag +1, half 1), same events (0.5 s bins, mean prediction x 0.5, clipped at 0, +0.5 s shift as lcv_feats.py),
same credit spec and lambda. Paired bootstrap vs the baseline preds of the same CV json.

  dino_cv2.py [--spec lin/all+dist+mlprior+fx:dino] [--lam 3e-5] [--dm]
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

T_ = Path(__file__).resolve().parent
sys.path.insert(0, str(T_))
sys.path.insert(0, str(T_.parents[2]))  # repo root, for autoscout
import dino_fit as DF  # noqa: E402
import fit_cv_launch as FC  # noqa: E402
from autoscout import attribution as A  # noqa: E402

L = DF.ROOT / "data" / "exp_launch"


def events(pred, t, team, dm, gmean):
    out = {}
    for tm in {int(x) for x in team if x >= 0}:
        mm = team == tm
        b = np.round(np.floor(t[mm] * 2) / 2, 2)
        ub, inv = np.unique(b, return_inverse=True)
        pb = np.bincount(inv, weights=pred[mm]) / np.bincount(inv)
        if dm:
            pb = pb - pb.mean() + gmean
        out[tm] = np.c_[ub + 0.25 + 0.5, np.maximum(pb, 0.0) * 0.5]  # +0.5: lcv_feats shift
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="lin/all+dist+mlprior+fx:dino")
    ap.add_argument("--lam", type=float, default=3e-5)
    ap.add_argument("--dm", action="store_true", help="demeaned variant (feature s_dinodm)")
    a = ap.parse_args()
    feats, recs = FC.load(str(L / "ds_launch.pkl"))
    fname = "s_dinodm" if a.dm else "s_dino"
    fi = feats.index(fname)
    sc = [r for r in recs if r.get("overlay") and "scout_n" in r and r.get("tba_total") and r["video_share"] is not None
          and r["scout_n"].sum() > 0 and len(r["n"]) > 0]
    keys_cv, g = FC.groups(sc)
    # ridge sufficient statistics per scouted match (as dino_fit.py)
    dkeys = sorted(p.stem for p in DF.DINO.glob("*.npz"))
    feat = ["box_inner", "box_cls"]
    raw = {k: DF.rows_of(k, feat, 1.0, 1.0) for k in dkeys}
    n = sum(np.isfinite(v[1]).sum() for v in raw.values())
    sx = sum(v[0][np.isfinite(v[1])].astype(np.float64).sum(0) for v in raw.values())
    sxx = sum((v[0][np.isfinite(v[1])].astype(np.float64) ** 2).sum(0) for v in raw.values())
    mu = sx / n; sd = np.sqrt(np.maximum(sxx / n - mu ** 2, 1e-8))
    Z, st = {}, {}
    for k, (X, y, t, team, _) in raw.items():
        Z[k] = np.c_[(X - mu) / sd, np.ones(len(X))].astype(np.float32)
        m = np.isfinite(y)
        Zm = Z[k][m].astype(np.float64)
        st[k] = (Zm.T @ Zm, Zm.T @ y[m])
        raw[k] = (None, y, t, team, None)  # free the float32 copy
    ZtZ = sum(s[0] for s in st.values()); Zty = sum(s[1] for s in st.values())
    R = 30.0 * np.eye(ZtZ.shape[0]); R[-1, -1] = 0.0
    gmean = Zty[-1] / ZtZ[-1, -1]
    spec = FC.parse_spec(a.spec)
    pred = np.zeros((len(sc), 3))
    for gi, m in enumerate(keys_cv):
        tr = []
        for r, gg in zip(sc, g):
            if gg == gi:
                continue
            j = r["key"]
            r2 = copy.copy(r)
            X = np.array(r["X"], copy=True)
            if j in st:
                w = np.linalg.solve(ZtZ - st[j][0] - (st[m][0] if m in st else 0) + R,
                                    Zty - st[j][1] - (st[m][1] if m in st else 0))
                ev = events(Z[j] @ w, raw[j][2], raw[j][3], a.dm, gmean)
                T = np.asarray(r["t"], float)
                for ri, team in enumerate(r["teams"]):
                    e = ev.get(int(team))
                    X[:, ri, fi] = (A.kernel(T[:, None] - e[None, :, 0]) * e[None, :, 1]).sum(1) if e is not None else 0.0
            r2["X"] = X
            tr.append(r2)
        th = FC.fit(spec, FC.Pack(tr, feats), a.lam)
        te = [i for i, gg in enumerate(g) if gg == gi]
        pred[te] = FC.predict(spec, th, FC.Pack([sc[i] for i in te], feats))
        print(f"fold {gi + 1}/{len(keys_cv)} {m}  g={th.get('g_' + fname[2:], th.get('lc_' + fname[2:]))}", flush=True)
    cv = json.loads((L / "cv_launch_lam3e-5.json").read_text())
    pb = np.array(cv["models"]["lin/all+dist+mlprior"]["pred"])
    p1 = np.array(cv["models"][a.spec]["pred"]) if a.spec in cv["models"] else None
    mb, mv = FC.metrics(pb, sc), FC.metrics(pred, sc)
    ci = FC.boot_diff(pred, pb, sc)
    print(f"\n{a.spec} double leave-out, lambda {a.lam:g}: TV {mv['tv']:.4f} corr {mv['corr']:.4f} top {mv['top']:.3f} "
          f"fuel MAE {mv['fuel_mae']:.2f}")
    for k in ("tv", "corr", "top", "fuel_mae"):
        print(f"   {k:8s} vs baseline {mv[k] - mb[k]:+.4f}  95% CI [{ci[k][0]:+.4f}, {ci[k][1]:+.4f}]")
    if p1 is not None:
        m1 = FC.metrics(p1, sc)
        print(f"   (single leave-out run: TV {m1['tv']:.4f}, fuel MAE {m1['fuel_mae']:.2f})")
    out = L / f"cv_dino2{'_dm' if a.dm else ''}.json"
    out.write_text(json.dumps({"spec": a.spec, "lam": a.lam, "metrics": mv, "ci_vs_base": ci, "pred": pred.tolist()}),
                   encoding="utf-8")


if __name__ == "__main__":
    main()
