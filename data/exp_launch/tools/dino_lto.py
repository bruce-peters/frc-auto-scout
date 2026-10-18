"""Leave-TEAMS-out DINO shooting ridge: does 'is shooting' generalise to robots the model never saw?

Same ridge as dino_fit.py (box_inner+box_cls, lam 30, lag +1, half 1), but for each scouted match the ridge is
trained only on labelled crops of teams NOT playing in it (sufficient statistics summed per team over all
matches), so it cannot use those robots' own appearance or a learned team prior - the situation at another event.
Writes "dinolto" / "dinolto_dm" events into data/exp_launch/sig_dino/<key>.pkl next to dino / dino_dm.
"""
import glob
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

T_ = Path(__file__).resolve().parent
sys.path.insert(0, str(T_))
import dino_fit as DF  # noqa: E402

FEAT = ["box_inner", "box_cls"]
TAG = os.environ.get("DINO_TAG", "dinolto")
keys = sorted(Path(p).stem for p in glob.glob(str(DF.SIG / "*.pkl")))
raw = {k: DF.rows_of(k, FEAT, 1.0, 1.0) for k in keys}
n = sum(np.isfinite(v[1]).sum() for v in raw.values())
sx = sum(v[0][np.isfinite(v[1])].astype(np.float64).sum(0) for v in raw.values())
sxx = sum((v[0][np.isfinite(v[1])].astype(np.float64) ** 2).sum(0) for v in raw.values())
mu = sx / n; sd = np.sqrt(np.maximum(sxx / n - mu ** 2, 1e-8))
team_st = defaultdict(lambda: [0.0, 0.0])
Z = {}
for k, (X, y, t, team, _) in raw.items():
    Z[k] = np.c_[(X - mu) / sd, np.ones(len(X))].astype(np.float32)
    m = np.isfinite(y)
    for tm in {int(x) for x in team[m]}:
        mm = m & (team == tm)
        Zm = Z[k][mm].astype(np.float64)
        team_st[tm][0] = team_st[tm][0] + Zm.T @ Zm
        team_st[tm][1] = team_st[tm][1] + Zm.T @ y[mm]
    raw[k] = (None, y, t, team, raw[k][4])
ZtZ = sum(v[0] for v in team_st.values()); Zty = sum(v[1] for v in team_st.values())
R = 30.0 * np.eye(ZtZ.shape[0]); R[-1, -1] = 0.0
P_all, Y_all = [], []
for k in keys:
    _, y, t, team, d = raw[k]
    playing = {int(x) for al in d["alliances"].values() for x in al["teams"]}
    A_ = ZtZ - sum((team_st[tm][0] for tm in playing if tm in team_st), 0.0)
    b_ = Zty - sum((team_st[tm][1] for tm in playing if tm in team_st), 0.0)
    w = np.linalg.solve(A_ + R, b_)
    gmean = b_[-1] / A_[-1, -1]
    pred = Z[k] @ w
    m = np.isfinite(y)
    P_all.append(pred[m]); Y_all.append(y[m])
    p = DF.OUT / f"{k}.pkl"
    with open(p, "rb") as fh:
        prev = pickle.load(fh)
    for tm in {int(x) for x in team if x >= 0}:
        mm = team == tm
        b = np.round(np.floor(t[mm] * 2) / 2, 2)
        ub, inv = np.unique(b, return_inverse=True)
        pb = np.bincount(inv, weights=pred[mm]) / np.bincount(inv)
        pdm = pb - pb.mean() + gmean
        prev.setdefault(tm, {}).update({
            TAG: [(float(x) + 0.25, float(max(v, 0.0)) * 0.5) for x, v in zip(ub, pb)],
            TAG + "_dm": [(float(x) + 0.25, float(max(v, 0.0)) * 0.5) for x, v in zip(ub, pdm)]})
    with open(p, "wb") as fh:
        pickle.dump(prev, fh)
P_all, Y_all = np.concatenate(P_all), np.concatenate(Y_all)
print(f"leave-teams-out ridge, {len(team_st)} teams: held-out per-crop corr {np.corrcoef(P_all, Y_all)[0, 1]:.3f}, "
      f"AUC(>=0.5 b/s) {DF.auc(P_all, Y_all >= 0.5):.3f}")
