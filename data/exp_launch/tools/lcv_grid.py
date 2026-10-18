"""prod_signals (grid series) -> per-jump features, appended to ds_launch.pkl -> data/exp_launch/ds_launch_g.pkl,
plus parity vs the event-based features the CV used (s_dinoc, s_drainraw from lcv_feats.py).

  s_dinocK / s_drainK   kernel form  sum(kernel(t - grid) * value)      (= what the CV'd features approximate)
  s_dinocW / s_drainW   window form  mean(value over [t - LAG_HI, t - LAG_LO])
dinoc uses the shipped ridge's leave-this-match-out weights for its 65 training matches (CV-safe).
Inputs: data/exp_launch/dino/<key>.npz (crops: t, view, v5 team, embeddings), v5_snapshot dets.json (fill, team),
result.json timeline (grid). No output/ reads.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\bruce\Documents\Coding\auto-scout")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "data" / "exp_launch")); sys.path.insert(0, str(ROOT / "data" / "exp_audit" / "tools"))
import prod_signals as PS  # noqa: E402
from common import TL  # noqa: E402

L, SNAP = ROOT / "data" / "exp_launch", ROOT / "data" / "exp_attrmodel" / "v5_snapshot"
ridge = PS.load_ridge(ROOT / "models" / "shooting_ridge.npz")
with open(L / "ds_launch.pkl", "rb") as fh:
    d = pickle.load(fh)
feats = list(d["feats"]); fi = {f: i for i, f in enumerate(feats)}
NEW = ["s_dinocK", "s_dinocW", "s_drainK", "s_drainW"]
cache, par = {}, {"dinoc": [], "drain": []}
for rec in d["recs"]:
    if "X" not in rec:
        continue
    k = rec["key"]
    X = np.asarray(rec["X"])
    if X.ndim != 3 or not len(X):
        rec["X"] = np.zeros((0, 3, len(feats) + len(NEW)))
        continue
    if k not in cache:
        r = json.loads((SNAP / k / "result.json").read_text())
        tl = TL(r)
        grid = np.arange(tl.t0(), tl.t1(), PS.GRID_STEP)
        rows = json.loads((SNAP / k / "dets.json").read_text())
        dr = PS.drain_core([x[0] for x in rows], [x[1] for x in rows], [np.nan if x[9] is None else x[9] for x in rows],
                           [-1 if x[7] is None else x[7] for x in rows], grid)
        dc = None
        p = L / "dino" / f"{k}.npz"
        if p.exists():
            E = np.load(p)
            Xe = np.concatenate([E[f].astype(np.float32) for f in PS.FEATS], 1)
            dc = PS.dinoc_core(E["t"].astype(float), E["view"].astype(int), E["team"].astype(int), Xe, ridge, grid,
                               exclude_key=k)
        cache = {k: (grid, dc, dr)}
    grid, dc, dr = cache[k]
    T = np.asarray(rec["t"], float)
    add = np.zeros((len(T), 3, len(NEW)))
    for ri, team in enumerate(rec["teams"]):
        vd = None if dc is None else dc.get(int(team))
        vr = dr.get(int(team))
        for ji, t in enumerate(T):
            if vd is not None:
                add[ji, ri, 0] = PS.jump_feature(grid, vd, t, "kernel"); add[ji, ri, 1] = PS.jump_feature(grid, vd, t, "window")
            if vr is not None:
                add[ji, ri, 2] = PS.jump_feature(grid, vr, t, "kernel"); add[ji, ri, 3] = PS.jump_feature(grid, vr, t, "window")
    if rec.get("overlay") and "scout_n" in rec:
        par["dinoc"].append((k, X[..., fi["s_dinoc"]].ravel(), add[..., 0].ravel()))
        par["drain"].append((k, X[..., fi["s_drainraw"]].ravel(), add[..., 2].ravel()))
    rec["X"] = np.concatenate([X, add], 2)
d["feats"] = feats + NEW
with open(L / "ds_launch_g.pkl", "wb") as fh:
    pickle.dump(d, fh)
print("parity: grid-series kernel features (prod_signals) vs the event-based features the CV used")
for nm, lst in par.items():
    a = np.concatenate([x[1] for x in lst]); b = np.concatenate([x[2] for x in lst])
    print(f"  {nm:6s} all scouted alliances: corr {np.corrcoef(a, b)[0, 1]:.5f}, median |diff| / mean {np.median(np.abs(a - b)) / max(a.mean(), 1e-9):.4f}, "
          f"sum ratio {b.sum() / max(a.sum(), 1e-9):.4f}")
    for k in ("2026cc_qm15", "2026cc_qm24", "2026cc_qm54"):
        sel = [x for x in lst if x[0] == k]
        if sel:
            a = np.concatenate([x[1] for x in sel]); b = np.concatenate([x[2] for x in sel])
            print(f"    {k}: {len(a)} robot-jumps, corr {np.corrcoef(a, b)[0, 1]:.5f}, max |diff| {np.abs(a - b).max():.3f} (mean value {a.mean():.3f})")
