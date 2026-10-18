"""'Is shooting' from appearance: ridge regression of the scouted launch rate on DINOv2 crop embeddings,
leave-one-match-out over the scouted Chezy quals.

Rows: crops (data/exp_launch/dino/<key>.npz) of robots with time-valid scouting. Target: scouted launches
(sig/<key>.pkl truth 'cont', weighted over scouters) in [t + LAG - HALF, t + LAG + HALF] per second.
Held-out predictions -> per-robot events for leval.py (0.5 s bins, mean prediction over the bin's crops x 0.5 s):
  dino      the prediction (clipped at 0)
  dino_dm   the same minus the robot's own match mean (+ the global mean): only the within-match variation, so a
            learned team prior cannot pass as 'shooting now'
Writes data/exp_launch/sig_dino/<key>.pkl and prints per-crop held-out corr / AUC.

  dino_fit.py [--feat box_inner,box_cls] [--lam 30] [--lag 1.0] [--half 1.0] [--tag dino]
"""
import argparse
import glob
import os
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\bruce\Documents\Coding\auto-scout")
DINO, SIG, OUT = ROOT / "data/exp_launch/dino", ROOT / "data/exp_launch/sig", ROOT / "data/exp_launch/sig_dino"


def auc(score, pos):
    from scipy.stats import rankdata
    r = rankdata(score)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def rows_of(k, feats, lag, half):
    E = np.load(DINO / f"{k}.npz")
    with open(SIG / f"{k}.pkl", "rb") as fh:
        d = pickle.load(fh)
    X = np.concatenate([E[f].astype(np.float32) for f in feats], 1)
    t, team = E["t"].astype(float), E["team"]
    if os.environ.get("DINO_CENTER"):  # remove identity: subtract each robot's mean embedding per view in this match
        vw = E["view"]
        for tm in {int(x) for x in team if x >= 0}:
            for v in np.unique(vw[team == tm]):
                mm = (team == tm) & (vw == v)
                X[mm] -= X[mm].mean(0)
    y = np.full(len(t), np.nan)
    timed = {tm for tm, n in d["timed"].items() if n}
    for tm in timed:
        tr = np.asarray(d["truth"][tm]["cont"], float).reshape(-1, 2)
        m = team == tm
        if not m.any():
            continue
        lo, hi = t[m] + lag - half, t[m] + lag + half
        if len(tr):
            ts = np.sort(tr[:, 0]); cw = np.r_[0, np.cumsum(tr[np.argsort(tr[:, 0]), 1])]
            y[m] = (cw[np.searchsorted(ts, hi)] - cw[np.searchsorted(ts, lo)]) / (2 * half)
        else:
            y[m] = 0.0
    return X, y, t, team, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="box_inner,box_cls")
    ap.add_argument("--lam", type=float, default=30.0)
    ap.add_argument("--lag", type=float, default=1.0)
    ap.add_argument("--half", type=float, default=1.0)
    ap.add_argument("--tag", default="dino")
    a = ap.parse_args()
    feats = a.feat.split(",")
    keys = sorted(Path(p).stem for p in glob.glob(str(DINO / "*.npz")) if (SIG / f"{Path(p).stem}.pkl").exists())
    # pass 1: per-match sufficient statistics on the labelled crops
    st, mom = {}, None
    for k in keys:
        X, y, *_ = rows_of(k, feats, a.lag, a.half)
        m = np.isfinite(y)
        Xm = X[m].astype(np.float64)
        s = {"n": m.sum(), "sx": Xm.sum(0), "sxx": (Xm ** 2).sum(0)}
        st[k] = s
        mom = s if mom is None else {kk: mom[kk] + s[kk] for kk in s}
    mu = mom["sx"] / mom["n"]; sd = np.sqrt(np.maximum(mom["sxx"] / mom["n"] - mu ** 2, 1e-8))
    for k in keys:
        X, y, *_ = rows_of(k, feats, a.lag, a.half)
        m = np.isfinite(y)
        Z = np.c_[(X[m] - mu) / sd, np.ones(m.sum())].astype(np.float64)
        st[k].update(ZtZ=Z.T @ Z, Zty=Z.T @ y[m])
    ZtZ = sum(s["ZtZ"] for s in st.values()); Zty = sum(s["Zty"] for s in st.values())
    D = ZtZ.shape[0]
    R = a.lam * np.eye(D); R[-1, -1] = 0.0
    OUT.mkdir(parents=True, exist_ok=True)
    P_all, Y_all, cor_dm = [], [], []
    gmean = Zty[-1] / ZtZ[-1, -1]
    for k in keys:
        w = np.linalg.solve(ZtZ - st[k]["ZtZ"] + R, Zty - st[k]["Zty"])
        X, y, t, team, d = rows_of(k, feats, a.lag, a.half)
        pred = np.c_[(X - mu) / sd, np.ones(len(X))] @ w
        m = np.isfinite(y)
        P_all.append(pred[m]); Y_all.append(y[m])
        out = {}
        for tm in {int(x) for x in team if x >= 0}:
            mm = team == tm
            b = np.round(np.floor(t[mm] * 2) / 2, 2)
            ub, inv = np.unique(b, return_inverse=True)
            pb = np.bincount(inv, weights=pred[mm]) / np.bincount(inv)
            pdm = pb - pb.mean() + gmean
            out[tm] = {a.tag: [(float(x) + 0.25, float(max(v, 0.0)) * 0.5) for x, v in zip(ub, pb)],
                       a.tag + "_dm": [(float(x) + 0.25, float(max(v, 0.0)) * 0.5) for x, v in zip(ub, pdm)]}
            if np.isfinite(y[mm]).all() and mm.sum() > 50 and np.std(y[mm]) > 0:
                yb = np.bincount(inv, weights=y[mm]) / np.bincount(inv)
                if np.std(yb) > 0 and np.std(pb) > 0:
                    cor_dm.append(np.corrcoef(pb, yb)[0, 1])
        prev = {}
        if (OUT / f"{k}.pkl").exists():
            with open(OUT / f"{k}.pkl", "rb") as fh:
                prev = pickle.load(fh)
        for tm, s in out.items():
            prev.setdefault(tm, {}).update(s)
        with open(OUT / f"{k}.pkl", "wb") as fh:
            pickle.dump(prev, fh)
    P_all, Y_all = np.concatenate(P_all), np.concatenate(Y_all)
    pos = Y_all >= 0.5
    print(f"{a.tag}: feats {feats} lam {a.lam:g} lag {a.lag:+g} half {a.half:g} | {len(Y_all)} labelled crops, "
          f"{pos.mean():.1%} with >= 0.5 balls/s | held-out per-crop corr {np.corrcoef(P_all, Y_all)[0, 1]:.3f}, "
          f"AUC(>=0.5 b/s) {auc(P_all, pos):.3f} | within robot-match corr over time (0.5 s bins) "
          f"mean {np.mean(cor_dm):.3f} (n={len(cor_dm)})")


if __name__ == "__main__":
    main()
