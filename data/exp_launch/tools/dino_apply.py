"""Full-data DINO shooting ridge (all scouted Chezy crops, settings of dino_fit.py) applied to every match that has
data/exp_launch/dino/<key>.npz -> data/exp_launch/sig_dino_full/<key>.pkl ({team: {"dino": ..., "dino_dm": ...}}),
and the weights -> data/exp_launch/dino_ridge.npz (mu, sd, w: the production artefact).
For the scouted matches use sig_dino/ (leave-one-match-out) in CV; sig_dino_full/ is for the other events.
"""
import glob
import os
import sys
import pickle
from pathlib import Path

import numpy as np

T_ = Path(__file__).resolve().parent
sys.path.insert(0, str(T_))
import dino_fit as DF  # noqa: E402

FEAT, LAM, LAG, HALF = ["box_inner", "box_cls"], 30.0, 1.0, 1.0
TAG = os.environ.get("DINO_TAG", "dino")
OUT = DF.ROOT / "data" / "exp_launch" / "sig_dino_full"
OUT.mkdir(parents=True, exist_ok=True)
scouted = sorted(Path(p).stem for p in glob.glob(str(DF.SIG / "*.pkl")))
n, sx, sxx = 0, 0.0, 0.0
for k in scouted:
    X, y, *_ = DF.rows_of(k, FEAT, LAG, HALF)
    m = np.isfinite(y)
    Xm = X[m].astype(np.float64)
    n += m.sum(); sx = sx + Xm.sum(0); sxx = sxx + (Xm ** 2).sum(0)
mu = sx / n; sd = np.sqrt(np.maximum(sxx / n - mu ** 2, 1e-8))
ZtZ, Zty = 0.0, 0.0
for k in scouted:
    X, y, *_ = DF.rows_of(k, FEAT, LAG, HALF)
    m = np.isfinite(y)
    Z = np.c_[(X[m] - mu) / sd, np.ones(m.sum())].astype(np.float64)
    ZtZ = ZtZ + Z.T @ Z; Zty = Zty + Z.T @ y[m]
R = LAM * np.eye(len(Zty)); R[-1, -1] = 0.0
w = np.linalg.solve(ZtZ + R, Zty)
gmean = Zty[-1] / ZtZ[-1, -1]
np.savez(DF.ROOT / "data" / "exp_launch" / f"{TAG}_ridge.npz", mu=mu.astype(np.float32), sd=sd.astype(np.float32),
         w=w.astype(np.float32), gmean=np.float32(gmean), feats=np.array(FEAT), lam=LAM, lag=LAG, half=HALF)
for p in sorted(glob.glob(str(DF.DINO / "*.npz"))):
    k = Path(p).stem
    E = np.load(p)
    X = np.concatenate([E[f].astype(np.float32) for f in FEAT], 1)
    t, team = E["t"].astype(float), E["team"]
    if os.environ.get("DINO_CENTER"):  # as dino_fit.rows_of
        vw = E["view"]
        for tm in {int(x) for x in team if x >= 0}:
            for v in np.unique(vw[team == tm]):
                mm = (team == tm) & (vw == v)
                X[mm] -= X[mm].mean(0)
    pred = np.c_[(X - mu) / sd, np.ones(len(X))] @ w
    out = {}
    for tm in {int(x) for x in team if x >= 0}:
        mm = team == tm
        b = np.round(np.floor(t[mm] * 2) / 2, 2)
        ub, inv = np.unique(b, return_inverse=True)
        pb = np.bincount(inv, weights=pred[mm]) / np.bincount(inv)
        pdm = pb - pb.mean() + gmean
        out[tm] = {TAG: [(float(x) + 0.25, float(max(v, 0.0)) * 0.5) for x, v in zip(ub, pb)],
                   TAG + "_dm": [(float(x) + 0.25, float(max(v, 0.0)) * 0.5) for x, v in zip(ub, pdm)]}
    prev = {}
    if (OUT / f"{k}.pkl").exists():
        with open(OUT / f"{k}.pkl", "rb") as fh:
            prev = pickle.load(fh)
    for tm_, s_ in out.items():
        prev.setdefault(tm_, {}).update(s_)
    with open(OUT / f"{k}.pkl", "wb") as fh:
        pickle.dump(prev, fh)
print(f"ridge on {n} labelled crops of {len(scouted)} scouted matches; applied to {len(glob.glob(str(DF.DINO / '*.npz')))} matches")
