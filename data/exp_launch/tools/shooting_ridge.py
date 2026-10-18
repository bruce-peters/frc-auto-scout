"""Build the shipped shooting ridge -> models/shooting_ridge.npz (and data/exp_launch/shooting_ridge.npz).

Training: every labelled crop of the 65 scouted Chezy quals (data/exp_launch/dino/<key>.npz: v5 DINO-identity team per
crop; 5026 scout taps, time-valid rows, launch = auto_start + tap - 0.8 s; label = scouted launches of that robot in
[t, t + 2 s] per second). x = box_inner + box_cls (768-d, frozen DINOv2 ViT-S/14), centred per robot, per view, within
the match (the robot's mean embedding in that view in that match subtracted); standardised (mu, sd over the
training crops); ridge lambda 30 (intercept unpenalised).
Fields: mu, sd (768), w (769, last = intercept), W_loo (65, 769) leave-one-match-out weights + loo_keys (for
CV datasets of the training matches only), train_keys, n_crops, lam, lag, half, feats, center.
"""
import glob
import os
import shutil
import sys
from pathlib import Path

import numpy as np

os.environ["DINO_CENTER"] = "1"
T_ = Path(__file__).resolve().parent
sys.path.insert(0, str(T_))
import dino_fit as DF  # noqa: E402

FEAT, LAM, LAG, HALF = ["box_inner", "box_cls"], 30.0, 1.0, 1.0
keys = sorted(Path(p).stem for p in glob.glob(str(DF.SIG / "*.pkl")) if (DF.DINO / f"{Path(p).stem}.npz").exists())
n, sx, sxx = 0, 0.0, 0.0
for k in keys:
    X, y, *_ = DF.rows_of(k, FEAT, LAG, HALF)
    m = np.isfinite(y)
    Xm = X[m].astype(np.float64)
    n += int(m.sum()); sx = sx + Xm.sum(0); sxx = sxx + (Xm ** 2).sum(0)
mu = sx / n; sd = np.sqrt(np.maximum(sxx / n - mu ** 2, 1e-8))
st = {}
for k in keys:
    X, y, *_ = DF.rows_of(k, FEAT, LAG, HALF)
    m = np.isfinite(y)
    Z = np.c_[(X[m] - mu) / sd, np.ones(m.sum())].astype(np.float64)
    st[k] = (Z.T @ Z, Z.T @ y[m])
ZtZ = sum(v[0] for v in st.values()); Zty = sum(v[1] for v in st.values())
R = LAM * np.eye(len(Zty)); R[-1, -1] = 0.0
w = np.linalg.solve(ZtZ + R, Zty)
W_loo = np.array([np.linalg.solve(ZtZ - st[k][0] + R, Zty - st[k][1]) for k in keys])
out = DF.ROOT / "data" / "exp_launch" / "shooting_ridge.npz"
np.savez(out, mu=mu.astype(np.float32), sd=sd.astype(np.float32), w=w.astype(np.float32),
         W_loo=W_loo.astype(np.float32), loo_keys=np.array(keys), train_keys=np.array(keys), n_crops=np.int64(n),
         lam=np.float32(LAM), lag=np.float32(LAG), half=np.float32(HALF), feats=np.array(FEAT),
         center=np.array("per robot, per view, within the match"))
shutil.copyfile(out, DF.ROOT / "models" / "shooting_ridge.npz")
d = np.load(DF.ROOT / "data" / "exp_launch" / "dinoc_ridge.npz")
print(f"{len(keys)} matches, {n} crops; full w equals dinoc_ridge.npz: max |dw| {np.abs(d['w'] - w).max():.2e}; "
      f"-> {out} and models/shooting_ridge.npz")
